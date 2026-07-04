// P3: Qwen3-ASR Thinker LLM (autoregressive text decoder) in C++.
// 28-layer Qwen3: RMSNorm + QK-norm + RoPE(theta1e6,hd128) + GQA(16/8) + causal + SwiGLU.
// tie_word_embeddings: lm_head = embed_tokens. Greedy. Recompute (no KV-cache yet).
// Starts from golden inputs_embeds (audio already injected), generates until eos.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>
using namespace std;
static const int D=1024, NH=16, NKV=8, HD=128, IM=3072, NL=28;
static const long VOCAB=151936;
static const float EPS=1e-6f, THETA=1e6f;
static const char* DIR="./convert_asr_thinker";

static vector<float> rd(const string&n,size_t cnt){
    FILE*f=fopen((string(DIR)+"/"+n+".f32").c_str(),"rb");
    if(!f){fprintf(stderr,"open %s\n",n.c_str());exit(1);}
    vector<float> v(cnt); size_t r=fread(v.data(),4,cnt,f); fclose(f);
    if(r!=cnt){fprintf(stderr,"read %s %zu/%zu\n",n.c_str(),r,cnt);exit(1);} return v;
}
struct Layer{ vector<float> iln,qw,kw,vw,ow,qn,kn,pln,gate,up,down; };

static void rmsnorm(const float*x,int Dm,const float*w,float*y){
    double s=0; for(int d=0;d<Dm;d++)s+=(double)x[d]*x[d];
    float inv=(float)(1.0/sqrt(s/Dm+EPS)); for(int d=0;d<Dm;d++) y[d]=x[d]*inv*w[d];
}
// y[N,O]=x[N,I]*W[O,I]^T
static void linear(const float*x,int N,int I,const vector<float>&W,int O,vector<float>&y){
    y.resize((size_t)N*O);
    #pragma omp parallel for
    for(int i=0;i<N;i++){ const float*xi=x+(size_t)i*I; float*yi=&y[(size_t)i*O];
        for(int o=0;o<O;o++){ const float*w=&W[(size_t)o*I]; double s=0; for(int k=0;k<I;k++)s+=xi[k]*w[k]; yi[o]=(float)s; } }
}
static inline float silu(float x){ return x/(1.0f+expf(-x)); }

int main(){
    // load weights
    vector<float> emb=rd("embed_tokens_weight",(size_t)VOCAB*D);
    vector<float> fnorm=rd("norm_weight",D);
    vector<Layer> Ls(NL);
    for(int l=0;l<NL;l++){ string p="layers_"+to_string(l)+"_"; Layer&L=Ls[l];
        L.iln=rd(p+"input_layernorm_weight",D);
        L.qw=rd(p+"self_attn_q_proj_weight",(size_t)NH*HD*D);
        L.kw=rd(p+"self_attn_k_proj_weight",(size_t)NKV*HD*D);
        L.vw=rd(p+"self_attn_v_proj_weight",(size_t)NKV*HD*D);
        L.ow=rd(p+"self_attn_o_proj_weight",(size_t)D*NH*HD);
        L.qn=rd(p+"self_attn_q_norm_weight",HD);
        L.kn=rd(p+"self_attn_k_norm_weight",HD);
        L.pln=rd(p+"post_attention_layernorm_weight",D);
        L.gate=rd(p+"mlp_gate_proj_weight",(size_t)IM*D);
        L.up=rd(p+"mlp_up_proj_weight",(size_t)IM*D);
        L.down=rd(p+"mlp_down_proj_weight",(size_t)D*IM);
    }
    // RoPE tables (inv_freq)
    vector<float> invf(HD/2); for(int i=0;i<HD/2;i++) invf[i]=powf(THETA,-(float)i/(HD/2));
    auto rope=[&](float*v,int pos){ // rotate_half RoPE on a head_dim vector
        for(int i=0;i<HD/2;i++){ float ang=pos*invf[i]; float c=cosf(ang),s=sinf(ang);
            float a=v[i], b=v[i+HD/2]; v[i]=a*c-b*s; v[i+HD/2]=b*c+a*s; }
    };

    int P=211; // prompt length (from inputs_embeds)
    vector<float> H=rd("inputs_embeds",(size_t)P*D); // (211,1024), will grow
    // golden gen ids
    vector<int> gen; { FILE*f=fopen("./convert_asr/gen_ids.i32","rb"); int v; while(fread(&v,4,1,f)==1)gen.push_back(v); fclose(f);}
    printf("prompt P=%d, golden total=%zu (gen text=%zu)\n",P,gen.size(),gen.size()-P);

    float scale=1.0f/sqrtf((float)HD);
    vector<int> out_tokens;
    int MAXGEN=60;
    for(int step=0; step<MAXGEN; step++){
        int L = P + step;                 // current seq length
        vector<float> x = H;              // copy (L rows)
        // run layers (recompute all L)
        for(int li=0; li<NL; li++){ Layer&LY=Ls[li];
            // attention
            vector<float> hn((size_t)L*D);
            for(int i=0;i<L;i++) rmsnorm(&x[(size_t)i*D],D,LY.iln.data(),&hn[(size_t)i*D]);
            vector<float> Q,K,V;
            linear(hn.data(),L,D,LY.qw,NH*HD,Q);
            linear(hn.data(),L,D,LY.kw,NKV*HD,K);
            linear(hn.data(),L,D,LY.vw,NKV*HD,V);
            // qk-norm per head + rope
            for(int i=0;i<L;i++){
                for(int h=0;h<NH;h++){ float*q=&Q[(size_t)i*NH*HD+h*HD]; float tmp[HD]; rmsnorm(q,HD,LY.qn.data(),tmp);
                    for(int d=0;d<HD;d++)q[d]=tmp[d]; rope(q,i); }
                for(int h=0;h<NKV;h++){ float*k=&K[(size_t)i*NKV*HD+h*HD]; float tmp[HD]; rmsnorm(k,HD,LY.kn.data(),tmp);
                    for(int d=0;d<HD;d++)k[d]=tmp[d]; rope(k,i); }
            }
            // causal GQA attention
            vector<float> ao((size_t)L*NH*HD,0.f);
            #pragma omp parallel for
            for(int i=0;i<L;i++){
                for(int h=0;h<NH;h++){ int kvh=h/(NH/NKV); const float*qi=&Q[(size_t)i*NH*HD+h*HD];
                    vector<float> sc(i+1); float mx=-1e30f;
                    for(int j=0;j<=i;j++){ const float*kj=&K[(size_t)j*NKV*HD+kvh*HD]; double d=0; for(int t=0;t<HD;t++)d+=qi[t]*kj[t]; float v=(float)d*scale; sc[j]=v; if(v>mx)mx=v; }
                    double sm=0; for(int j=0;j<=i;j++){ sc[j]=expf(sc[j]-mx); sm+=sc[j]; }
                    float inv=(float)(1.0/sm); float*aoi=&ao[(size_t)i*NH*HD+h*HD];
                    for(int j=0;j<=i;j++){ float w=sc[j]*inv; const float*vj=&V[(size_t)j*NKV*HD+kvh*HD]; for(int t=0;t<HD;t++)aoi[t]+=w*vj[t]; }
                }
            }
            vector<float> attn; linear(ao.data(),L,NH*HD,LY.ow,D,attn);
            for(size_t i=0;i<x.size();i++) x[i]+=attn[i];
            // MLP
            vector<float> h2((size_t)L*D);
            for(int i=0;i<L;i++) rmsnorm(&x[(size_t)i*D],D,LY.pln.data(),&h2[(size_t)i*D]);
            vector<float> g,u; linear(h2.data(),L,D,LY.gate,IM,g); linear(h2.data(),L,D,LY.up,IM,u);
            for(size_t i=0;i<g.size();i++) g[i]=silu(g[i])*u[i];
            vector<float> dn; linear(g.data(),L,IM,LY.down,D,dn);
            for(size_t i=0;i<x.size();i++) x[i]+=dn[i];
        }
        // final norm on last position, lm_head argmax
        float hn[D]; rmsnorm(&x[(size_t)(L-1)*D],D,fnorm.data(),hn);
        long best=-1; double bestv=-1e30;
        #pragma omp parallel
        { long lb=-1; double lv=-1e30;
          #pragma omp for nowait
          for(long v=0;v<VOCAB;v++){ const float*e=&emb[(size_t)v*D]; double s=0; for(int d=0;d<D;d++)s+=hn[d]*e[d]; if(s>lv){lv=s;lb=v;} }
          #pragma omp critical
          { if(lv>bestv){bestv=lv;best=lb;} }
        }
        int tok=(int)best;
        int gold = (P+step<(int)gen.size())? gen[P+step] : -1;
        printf("step %d: tok=%d gold=%d %s\n",step,tok,gold, tok==gold?"OK":"XX");
        out_tokens.push_back(tok);
        if(tok==151643||tok==151645) break;
        // append embedding
        H.insert(H.end(), &emb[(size_t)tok*D], &emb[(size_t)tok*D]+D);
    }
    // score
    int ok=0,tot=0; for(size_t i=0;i<out_tokens.size();i++){ int g=(P+(int)i<(int)gen.size())?gen[P+i]:-1; if(out_tokens[i]==g)ok++; tot++; }
    printf("match %d/%d\n",ok,tot);
    return 0;
}
