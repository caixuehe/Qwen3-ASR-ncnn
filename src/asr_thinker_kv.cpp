// P5: Qwen3-ASR Thinker with KV-cache. Prefill P prompt rows (cache K/V), then decode
// one token at a time attending against the cache. O(T) per step vs O(T^2) recompute.
// Verifies generated tokens == golden (52/52).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
using namespace std;
static const int D=1024,NH=16,NKV=8,HD=128,IM=3072,NL=28; static const long VOCAB=151936;
static const char* DIR="./convert_asr_thinker";
static vector<float> rd(const string&n,size_t c){FILE*f=fopen((string(DIR)+"/"+n+".f32").c_str(),"rb");if(!f){fprintf(stderr,"open %s\n",n.c_str());exit(1);}vector<float>v(c);size_t r=fread(v.data(),4,c,f);fclose(f);if(r!=c){fprintf(stderr,"read %s\n",n.c_str());exit(1);}return v;}
static void rmsnorm(const float*x,int Dm,const float*w,float*y,float eps=1e-6f){double s=0;for(int d=0;d<Dm;d++)s+=(double)x[d]*x[d];float iv=(float)(1.0/sqrt(s/Dm+eps));for(int d=0;d<Dm;d++)y[d]=x[d]*iv*w[d];}
static vector<float> lin(const float*x,int N,int I,const vector<float>&W,int O){vector<float>y((size_t)N*O);
    #pragma omp parallel for
    for(int i=0;i<N;i++){const float*xi=x+(size_t)i*I;float*yi=&y[(size_t)i*O];for(int o=0;o<O;o++){const float*w=&W[(size_t)o*I];double s=0;for(int k=0;k<I;k++)s+=xi[k]*w[k];yi[o]=(float)s;}}return y;}
static inline float silu(float x){return x/(1.0f+expf(-x));}
struct TL{vector<float> iln,qw,kw,vw,ow,qn,kn,pln,gate,up,down;};

int main(){
    auto fnorm=rd("norm_weight",D);vector<TL> Ls(NL);
    for(int l=0;l<NL;l++){string p="layers_"+to_string(l)+"_";TL&L=Ls[l];
        L.iln=rd(p+"input_layernorm_weight",D);L.qw=rd(p+"self_attn_q_proj_weight",(size_t)NH*HD*D);L.kw=rd(p+"self_attn_k_proj_weight",(size_t)NKV*HD*D);
        L.vw=rd(p+"self_attn_v_proj_weight",(size_t)NKV*HD*D);L.ow=rd(p+"self_attn_o_proj_weight",(size_t)D*NH*HD);L.qn=rd(p+"self_attn_q_norm_weight",HD);L.kn=rd(p+"self_attn_k_norm_weight",HD);
        L.pln=rd(p+"post_attention_layernorm_weight",D);L.gate=rd(p+"mlp_gate_proj_weight",(size_t)IM*D);L.up=rd(p+"mlp_up_proj_weight",(size_t)IM*D);L.down=rd(p+"mlp_down_proj_weight",(size_t)D*IM);}
    vector<float> invf(HD/2);for(int i=0;i<HD/2;i++)invf[i]=powf(1e6f,-(float)i/(HD/2));
    auto rope=[&](float*v,int pos){for(int i=0;i<HD/2;i++){float a=pos*invf[i],c=cosf(a),s=sinf(a);float x0=v[i],y0=v[i+HD/2];v[i]=x0*c-y0*s;v[i+HD/2]=y0*c+x0*s;}};
    auto emb=rd("embed_tokens_weight",(size_t)VOCAB*D);
    int P=211;vector<float> ie=rd("inputs_embeds",(size_t)P*D);
    vector<int> gen;{FILE*f=fopen("./convert_asr/gen_ids.i32","rb");int v;while(fread(&v,4,1,f)==1)gen.push_back(v);fclose(f);}
    float scale=1.0f/sqrtf((float)HD);
    vector<vector<float>> kc(NL),vc(NL); // caches: rows of NKV*HD

    // process M new rows at abs positions pos0.., update x in place; returns x (M x D)
    auto run=[&](vector<float> x,int M,int pos0)->vector<float>{
        for(int l=0;l<NL;l++){TL&LY=Ls[l];
            vector<float> hn((size_t)M*D);for(int i=0;i<M;i++)rmsnorm(&x[(size_t)i*D],D,LY.iln.data(),&hn[(size_t)i*D]);
            auto Q=lin(hn.data(),M,D,LY.qw,NH*HD);auto K=lin(hn.data(),M,D,LY.kw,NKV*HD);auto V=lin(hn.data(),M,D,LY.vw,NKV*HD);
            for(int i=0;i<M;i++){int p=pos0+i;
                for(int h=0;h<NH;h++){float*q=&Q[(size_t)i*NH*HD+h*HD];float t[128];rmsnorm(q,HD,LY.qn.data(),t);for(int d=0;d<HD;d++)q[d]=t[d];rope(q,p);}
                for(int h=0;h<NKV;h++){float*k=&K[(size_t)i*NKV*HD+h*HD];float t[128];rmsnorm(k,HD,LY.kn.data(),t);for(int d=0;d<HD;d++)k[d]=t[d];rope(k,p);}}
            // append K,V to cache
            kc[l].insert(kc[l].end(),K.begin(),K.end());vc[l].insert(vc[l].end(),V.begin(),V.end());
            int clen=pos0+M;
            vector<float> ao((size_t)M*NH*HD,0.f);
            #pragma omp parallel for
            for(int i=0;i<M;i++){int p=pos0+i;const float*Qi=&Q[(size_t)i*NH*HD];
                for(int h=0;h<NH;h++){int kvh=h/(NH/NKV);const float*qi=&Qi[h*HD];vector<float> sc(p+1);float mx=-1e30f;
                    for(int j=0;j<=p;j++){const float*kj=&kc[l][(size_t)j*NKV*HD+kvh*HD];double d=0;for(int t=0;t<HD;t++)d+=qi[t]*kj[t];float v=(float)d*scale;sc[j]=v;if(v>mx)mx=v;}
                    double sm=0;for(int j=0;j<=p;j++){sc[j]=expf(sc[j]-mx);sm+=sc[j];}float iv=(float)(1.0/sm);float*aoi=&ao[(size_t)i*NH*HD+h*HD];
                    for(int j=0;j<=p;j++){float w=sc[j]*iv;const float*vj=&vc[l][(size_t)j*NKV*HD+kvh*HD];for(int t=0;t<HD;t++)aoi[t]+=w*vj[t];}}}
            auto attn=lin(ao.data(),M,NH*HD,LY.ow,D);for(size_t i=0;i<x.size();i++)x[i]+=attn[i];
            vector<float> h2((size_t)M*D);for(int i=0;i<M;i++)rmsnorm(&x[(size_t)i*D],D,LY.pln.data(),&h2[(size_t)i*D]);
            auto g=lin(h2.data(),M,D,LY.gate,IM);auto u=lin(h2.data(),M,D,LY.up,IM);for(size_t i=0;i<g.size();i++)g[i]=silu(g[i])*u[i];
            auto dn=lin(g.data(),M,IM,LY.down,D);for(size_t i=0;i<x.size();i++)x[i]+=dn[i];
        }
        return x;
    };
    auto argmax=[&](const float*x)->int{float hn[1024];rmsnorm(x,D,fnorm.data(),hn);long best=-1;double bv=-1e30;
        #pragma omp parallel
        {long lb=-1;double lv=-1e30;
         #pragma omp for nowait
         for(long v=0;v<VOCAB;v++){const float*e=&emb[(size_t)v*D];double s=0;for(int d=0;d<D;d++)s+=hn[d]*e[d];if(s>lv){lv=s;lb=v;}}
         #pragma omp critical
         {if(lv>bv){bv=lv;best=lb;}}}return (int)best;};

    clock_t t0=clock();
    // prefill
    auto xp=run(ie,P,0);
    int tok=argmax(&xp[(size_t)(P-1)*D]);
    vector<int> out; out.push_back(tok);
    int L=P;
    for(int step=1; step<448 && tok!=151645 && tok!=151643; step++){
        vector<float> row(&emb[(size_t)tok*D],&emb[(size_t)tok*D]+D);
        auto xr=run(row,1,L); L++;
        tok=argmax(xr.data()); out.push_back(tok);
    }
    double sec=double(clock()-t0)/CLOCKS_PER_SEC;
    int ok=0;for(size_t i=0;i<out.size();i++){int g=(P+(int)i<(int)gen.size())?gen[P+i]:-1;if(out[i]==g)ok++;}
    printf("KV-cache: generated %zu tokens, match %d/%zu, %.1fs (cpu)\n",out.size(),ok,out.size(),sec);
    return 0;
}
