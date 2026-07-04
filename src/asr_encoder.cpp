// P2-B: Qwen3-ASR audio encoder transformer (enc_in -> audio_embed) in C++.
// 18 pre-LN layers, MHA (14 heads, head_dim 64, bias), block-diagonal bidirectional
// attention over cu_seqlens blocks, gelu MLP, LayerNorm; then ln_post + proj1/gelu/proj2.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
using namespace std;
static const int D=896, H=14, HD=64, FF=3584, OUT=1024, NL=18;
static const char* DIR="./convert_asr_enc";

static vector<float> rd(const string&n,size_t cnt){
    FILE*f=fopen((string(DIR)+"/"+n+".f32").c_str(),"rb");
    if(!f){fprintf(stderr,"open %s\n",n.c_str());exit(1);}
    vector<float> v(cnt); size_t r=fread(v.data(),4,cnt,f); fclose(f);
    if(r!=cnt){fprintf(stderr,"read %s %zu/%zu\n",n.c_str(),r,cnt);exit(1);} return v;
}
static void layernorm(float*x,int N,int Dm,const float*w,const float*b){
    for(int i=0;i<N;i++){ float*p=x+(size_t)i*Dm; double mu=0; for(int d=0;d<Dm;d++)mu+=p[d]; mu/=Dm;
        double var=0; for(int d=0;d<Dm;d++){double t=p[d]-mu;var+=t*t;} var/=Dm;
        double inv=1.0/sqrt(var+1e-5); for(int d=0;d<Dm;d++) p[d]=(float)((p[d]-mu)*inv)*w[d]+b[d]; }
}
// y[N,O] = x[N,I] * W[O,I]^T + b
static vector<float> linear(const vector<float>&x,int N,int I,const vector<float>&W,const vector<float>&b,int O){
    vector<float> y((size_t)N*O);
    for(int i=0;i<N;i++){ const float*xi=&x[(size_t)i*I]; float*yi=&y[(size_t)i*O];
        for(int o=0;o<O;o++){ const float*w=&W[(size_t)o*I]; double s=b.empty()?0:b[o];
            for(int k=0;k<I;k++) s+=xi[k]*w[k]; yi[o]=(float)s; } }
    return y;
}
static inline float gelu(float x){ return 0.5f*x*(1.0f+erff(x*0.70710678f)); }

int main(){
    // determine N from cu_seqlens last value
    FILE*cf=fopen((string(DIR)+"/cu_seqlens.i32").c_str(),"rb");
    vector<int> cu; { int v; while(fread(&v,4,1,cf)==1) cu.push_back(v);} fclose(cf);
    int N=cu.back();
    printf("N=%d cu=[",N); for(int c:cu)printf("%d,",c); printf("]\n");
    bool FULL = getenv("FULL")!=nullptr;   // FULL=1 -> full attention (ignore cu blocks)
    vector<int> blk = FULL ? vector<int>{0,N} : cu;
    printf("attention mode: %s\n", FULL?"FULL":"block-diagonal");
    vector<float> x = rd("enc_in",(size_t)N*D);
    float scale=1.0f/sqrtf((float)HD);

    for(int L=0;L<NL;L++){
        string p="layers_"+to_string(L)+"_";
        auto lnw=rd(p+"self_attn_layer_norm_weight",D), lnb=rd(p+"self_attn_layer_norm_bias",D);
        auto qw=rd(p+"self_attn_q_proj_weight",(size_t)D*D), qb=rd(p+"self_attn_q_proj_bias",D);
        auto kw=rd(p+"self_attn_k_proj_weight",(size_t)D*D), kb=rd(p+"self_attn_k_proj_bias",D);
        auto vw=rd(p+"self_attn_v_proj_weight",(size_t)D*D), vb=rd(p+"self_attn_v_proj_bias",D);
        auto ow=rd(p+"self_attn_out_proj_weight",(size_t)D*D), ob=rd(p+"self_attn_out_proj_bias",D);
        auto fnw=rd(p+"final_layer_norm_weight",D), fnb=rd(p+"final_layer_norm_bias",D);
        auto f1w=rd(p+"fc1_weight",(size_t)FF*D), f1b=rd(p+"fc1_bias",FF);
        auto f2w=rd(p+"fc2_weight",(size_t)D*FF), f2b=rd(p+"fc2_bias",D);

        // --- attention block ---
        vector<float> h=x;                         // residual copy
        layernorm(h.data(),N,D,lnw.data(),lnb.data());
        auto Q=linear(h,N,D,qw,qb,D), K=linear(h,N,D,kw,kb,D), V=linear(h,N,D,vw,vb,D);
        vector<float> ao((size_t)N*D,0.f);
        // block-diagonal bidirectional attention
        for(size_t bi=0;bi+1<blk.size();bi++){
            int s=blk[bi], e=blk[bi+1];
            for(int hh=0;hh<H;hh++){ int off=hh*HD;
                for(int i=s;i<e;i++){
                    const float*qi=&Q[(size_t)i*D+off];
                    // scores
                    vector<float> sc(e-s); float mx=-1e30f;
                    for(int j=s;j<e;j++){ const float*kj=&K[(size_t)j*D+off]; double d=0;
                        for(int t=0;t<HD;t++)d+=qi[t]*kj[t]; float v=(float)d*scale; sc[j-s]=v; if(v>mx)mx=v; }
                    double sum=0; for(int j=0;j<e-s;j++){ sc[j]=expf(sc[j]-mx); sum+=sc[j]; }
                    float inv=(float)(1.0/sum); float*aoi=&ao[(size_t)i*D+off];
                    for(int j=s;j<e;j++){ float wgt=sc[j-s]*inv; const float*vj=&V[(size_t)j*D+off];
                        for(int t=0;t<HD;t++) aoi[t]+=wgt*vj[t]; }
                }
            }
        }
        auto attn=linear(ao,N,D,ow,ob,D);
        for(size_t i=0;i<x.size();i++) x[i]+=attn[i];
        // --- MLP block ---
        vector<float> h2=x;
        layernorm(h2.data(),N,D,fnw.data(),fnb.data());
        auto u=linear(h2,N,D,f1w,f1b,FF);
        for(auto&z:u) z=gelu(z);
        auto ff=linear(u,N,FF,f2w,f2b,D);
        for(size_t i=0;i<x.size();i++) x[i]+=ff[i];
        if(L==0){ auto g0=rd("layer0_out",(size_t)N*D); double md=0,ss=0,gg=0;
            for(size_t i=0;i<x.size();i++){double d=x[i]-g0[i];if(fabs(d)>md)md=fabs(d);ss+=d*d;gg+=g0[i]*g0[i];}
            printf("layer0_out vs golden: maxdiff=%.3e relRMS=%.3e\n",md,sqrt(ss/gg)); }
    }
    // head: ln_post -> proj1 -> gelu -> proj2
    auto lpw=rd("ln_post_weight",D), lpb=rd("ln_post_bias",D);
    layernorm(x.data(),N,D,lpw.data(),lpb.data());
    auto p1w=rd("proj1_weight",(size_t)D*D), p1b=rd("proj1_bias",D);
    auto p2w=rd("proj2_weight",(size_t)OUT*D), p2b=rd("proj2_bias",OUT);
    auto y=linear(x,N,D,p1w,p1b,D);
    for(auto&z:y) z=gelu(z);
    auto out=linear(y,N,D,p2w,p2b,OUT);

    // compare to golden audio_embed (in ../convert_asr)
    FILE*gf=fopen("./convert_asr/audio_embed.f32","rb");
    vector<float> g((size_t)N*OUT); size_t r=fread(g.data(),4,g.size(),gf); fclose(gf);
    double md=0,ss=0,gg=0; for(size_t i=0;i<out.size();i++){double d=out[i]-g[i];if(fabs(d)>md)md=fabs(d);ss+=d*d;gg+=g[i]*g[i];}
    printf("audio_embed (%d,%d) vs golden: maxdiff=%.3e relRMS=%.3e (read %zu)\n",N,OUT,md,sqrt(ss/gg),r);
    return 0;
}
