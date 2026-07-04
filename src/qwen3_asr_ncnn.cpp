// Qwen3-ASR end-to-end: audio waveform -> text, pure C++ (P1..P4 chained).
// mel (Whisper) -> conv2d frontend -> 18-layer audio encoder -> prompt+audio inject
// -> 28-layer Thinker LLM (greedy) -> byte-level BPE decode.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>
#include "net.h"
using namespace std;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static string ENC="./convert_asr_enc", THK="./convert_asr_thinker", ROOT="./convert_asr", TOK="./convert_asr_tok";
static vector<float> rdf(const string&p,size_t n){ FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(1);}
    vector<float> v(n); size_t r=fread(v.data(),4,n,f); fclose(f); if(r!=n){fprintf(stderr,"read %s %zu/%zu\n",p.c_str(),r,n);exit(1);} return v; }
static vector<float> E(const string&n,size_t c){return rdf(ENC+"/"+n+".f32",c);}
static vector<float> Tk(const string&n,size_t c){return rdf(THK+"/"+n+".f32",c);}

static inline float gelu(float x){return 0.5f*x*(1.0f+erff(x*0.70710678f));}
static inline float silu(float x){return x/(1.0f+expf(-x));}
static inline int convlen(int x){return (x-1)/2+1;}
static void layernorm(float*x,int N,int Dm,const float*w,const float*b){
    for(int i=0;i<N;i++){float*p=x+(size_t)i*Dm;double mu=0;for(int d=0;d<Dm;d++)mu+=p[d];mu/=Dm;
        double var=0;for(int d=0;d<Dm;d++){double t=p[d]-mu;var+=t*t;}var/=Dm;double inv=1.0/sqrt(var+1e-5);
        for(int d=0;d<Dm;d++)p[d]=(float)((p[d]-mu)*inv)*w[d]+b[d];}}
static void rmsnorm(const float*x,int Dm,const float*w,float*y,float eps=1e-6f){
    double s=0;for(int d=0;d<Dm;d++)s+=(double)x[d]*x[d];float inv=(float)(1.0/sqrt(s/Dm+eps));
    for(int d=0;d<Dm;d++)y[d]=x[d]*inv*w[d];}
static vector<float> lin(const float*x,int N,int I,const vector<float>&W,const vector<float>*b,int O){
    vector<float> y((size_t)N*O);
    #pragma omp parallel for
    for(int i=0;i<N;i++){const float*xi=x+(size_t)i*I;float*yi=&y[(size_t)i*O];
        for(int o=0;o<O;o++){const float*w=&W[(size_t)o*I];double s=b?(*b)[o]:0;for(int k=0;k<I;k++)s+=xi[k]*w[k];yi[o]=(float)s;}}
    return y;}

// ---------- P1: mel ----------
static vector<float> mel_extract(const vector<float>&wav){
    const int NFFT=400,HOP=160,NMEL=128,NFREQ=201;
    auto filt=rdf(ROOT+"/mel_filters.f32",(size_t)NFREQ*NMEL);
    long wl=wav.size(),pad=NFFT/2,PL=wl+2*pad; vector<float> x(PL);
    for(long i=0;i<PL;i++){long j=i-pad;if(j<0)j=-j;else if(j>=wl)j=2*wl-2-j;x[i]=wav[j];}
    long T=(PL-NFFT)/HOP;
    vector<float> win(NFFT);for(int n=0;n<NFFT;n++)win[n]=0.5f-0.5f*cosf(2*M_PI*n/NFFT);
    vector<float> C((size_t)NFREQ*NFFT),S((size_t)NFREQ*NFFT);
    for(int k=0;k<NFREQ;k++)for(int n=0;n<NFFT;n++){double a=-2*M_PI*k*n/NFFT;C[(size_t)k*NFFT+n]=cos(a);S[(size_t)k*NFFT+n]=sin(a);}
    vector<float> mel((size_t)NMEL*T);
    #pragma omp parallel for
    for(long t=0;t<T;t++){ float fr[400],pw[201]; long off=t*HOP;
        for(int n=0;n<NFFT;n++)fr[n]=x[off+n]*win[n];
        for(int k=0;k<NFREQ;k++){const float*ck=&C[(size_t)k*NFFT];const float*sk=&S[(size_t)k*NFFT];double re=0,im=0;
            for(int n=0;n<NFFT;n++){re+=fr[n]*ck[n];im+=fr[n]*sk[n];}pw[k]=(float)(re*re+im*im);}
        for(int m=0;m<NMEL;m++){double s=0;for(int k=0;k<NFREQ;k++)s+=(double)filt[(size_t)k*NMEL+m]*pw[k];
            mel[(size_t)m*T+t]=(float)log10(s<1e-10?1e-10:s);}}
    float gmax=-1e30f;for(size_t i=0;i<mel.size();i++)if(mel[i]>gmax)gmax=mel[i];float fl=gmax-8;
    for(size_t i=0;i<mel.size();i++){float v=mel[i]<fl?fl:mel[i];mel[i]=(v+4)/4;}
    return mel; // (128,T)
}

// ---------- P2-A: conv frontend via ncnn ----------
static ncnn::Net g_fe;
static vector<float> frontend(const vector<float>&mel,int T,int&N){
    const int NMEL=128,WIN=100,D=896;
    auto pe=E("positional_embedding",(size_t)1500*D);
    vector<int> cl;int rem=T;while(rem>0){int c=rem>=WIN?WIN:rem;cl.push_back(c);rem-=c;}
    vector<float> ein;
    for(size_t ci=0;ci<cl.size();ci++){
        ncnn::Mat in(WIN,NMEL,1); in.fill(0.f); float*p=in.channel(0); int base=(int)ci*WIN;
        for(int f=0;f<NMEL;f++)for(int t=0;t<WIN;t++){int tt=base+t;if(tt<T)p[f*WIN+t]=mel[(size_t)f*T+tt];}
        ncnn::Extractor ex=g_fe.create_extractor(); ex.input("in0",in); ncnn::Mat out; ex.extract("out0",out);
        int valid=convlen(convlen(convlen(cl[ci])));
        for(int t=0;t<valid;t++){const float*r=out.row(t); for(int d=0;d<D;d++)ein.push_back(r[d]+pe[(size_t)t*D+d]);}
    }
    N=ein.size()/D;return ein;}

// ---------- P2-B: encoder transformer ----------
static vector<float> encoder(vector<float> x,int N){
    const int D=896,H=14,HD=64,FF=3584,OUT=1024,NL=18;float scale=1.f/sqrtf(HD);
    for(int L=0;L<NL;L++){string p="layers_"+to_string(L)+"_";
        auto lnw=E(p+"self_attn_layer_norm_weight",D),lnb=E(p+"self_attn_layer_norm_bias",D);
        auto qw=E(p+"self_attn_q_proj_weight",(size_t)D*D),qb=E(p+"self_attn_q_proj_bias",D);
        auto kw=E(p+"self_attn_k_proj_weight",(size_t)D*D),kb=E(p+"self_attn_k_proj_bias",D);
        auto vw=E(p+"self_attn_v_proj_weight",(size_t)D*D),vb=E(p+"self_attn_v_proj_bias",D);
        auto ow=E(p+"self_attn_out_proj_weight",(size_t)D*D),ob=E(p+"self_attn_out_proj_bias",D);
        auto fnw=E(p+"final_layer_norm_weight",D),fnb=E(p+"final_layer_norm_bias",D);
        auto f1w=E(p+"fc1_weight",(size_t)FF*D),f1b=E(p+"fc1_bias",FF);auto f2w=E(p+"fc2_weight",(size_t)D*FF),f2b=E(p+"fc2_bias",D);
        vector<float> h=x;layernorm(h.data(),N,D,lnw.data(),lnb.data());
        auto Q=lin(h.data(),N,D,qw,&qb,D),K=lin(h.data(),N,D,kw,&kb,D),V=lin(h.data(),N,D,vw,&vb,D);
        vector<float> ao((size_t)N*D,0.f);
        #pragma omp parallel for
        for(int i=0;i<N;i++)for(int hh=0;hh<H;hh++){int off=hh*HD;const float*qi=&Q[(size_t)i*D+off];
            vector<float> sc(N);float mx=-1e30f;for(int j=0;j<N;j++){const float*kj=&K[(size_t)j*D+off];double d=0;for(int t=0;t<HD;t++)d+=qi[t]*kj[t];float v=(float)d*scale;sc[j]=v;if(v>mx)mx=v;}
            double sm=0;for(int j=0;j<N;j++){sc[j]=expf(sc[j]-mx);sm+=sc[j];}float iv=(float)(1.0/sm);float*aoi=&ao[(size_t)i*D+off];
            for(int j=0;j<N;j++){float w=sc[j]*iv;const float*vj=&V[(size_t)j*D+off];for(int t=0;t<HD;t++)aoi[t]+=w*vj[t];}}
        auto attn=lin(ao.data(),N,D,ow,&ob,D);for(size_t i=0;i<x.size();i++)x[i]+=attn[i];
        vector<float> h2=x;layernorm(h2.data(),N,D,fnw.data(),fnb.data());auto u=lin(h2.data(),N,D,f1w,&f1b,FF);for(auto&z:u)z=gelu(z);auto ff=lin(u.data(),N,FF,f2w,&f2b,D);for(size_t i=0;i<x.size();i++)x[i]+=ff[i];
    }
    auto lpw=E("ln_post_weight",D),lpb=E("ln_post_bias",D);layernorm(x.data(),N,D,lpw.data(),lpb.data());
    auto p1w=E("proj1_weight",(size_t)D*D),p1b=E("proj1_bias",D),p2w=E("proj2_weight",(size_t)OUT*D),p2b=E("proj2_bias",OUT);
    auto y=lin(x.data(),N,D,p1w,&p1b,D);for(auto&z:y)z=gelu(z);return lin(y.data(),N,D,p2w,&p2b,OUT); // (N,1024)
}

// ---------- P3: Thinker (recompute greedy) ----------
struct TL{vector<float> iln,qw,kw,vw,ow,qn,kn,pln,gate,up,down;};
static vector<int> thinker(vector<float> H,int P,const vector<float>&emb){
    const int D=1024,NH=16,NKV=8,HD=128,IM=3072,NL=28;const long VOCAB=151936;float scale=1.f/sqrtf(HD);
    auto fnorm=Tk("norm_weight",D);vector<TL> Ls(NL);
    for(int l=0;l<NL;l++){string p="layers_"+to_string(l)+"_";TL&L=Ls[l];
        L.iln=Tk(p+"input_layernorm_weight",D);L.qw=Tk(p+"self_attn_q_proj_weight",(size_t)NH*HD*D);L.kw=Tk(p+"self_attn_k_proj_weight",(size_t)NKV*HD*D);
        L.vw=Tk(p+"self_attn_v_proj_weight",(size_t)NKV*HD*D);L.ow=Tk(p+"self_attn_o_proj_weight",(size_t)D*NH*HD);L.qn=Tk(p+"self_attn_q_norm_weight",HD);L.kn=Tk(p+"self_attn_k_norm_weight",HD);
        L.pln=Tk(p+"post_attention_layernorm_weight",D);L.gate=Tk(p+"mlp_gate_proj_weight",(size_t)IM*D);L.up=Tk(p+"mlp_up_proj_weight",(size_t)IM*D);L.down=Tk(p+"mlp_down_proj_weight",(size_t)D*IM);}
    vector<float> invf(HD/2);for(int i=0;i<HD/2;i++)invf[i]=powf(1e6f,-(float)i/(HD/2));
    auto rope=[&](float*v,int pos){for(int i=0;i<HD/2;i++){float a=pos*invf[i],c=cosf(a),s=sinf(a);float x0=v[i],y0=v[i+HD/2];v[i]=x0*c-y0*s;v[i+HD/2]=y0*c+x0*s;}};
    vector<int> out;int MAXGEN=448;
    for(int step=0;step<MAXGEN;step++){int L=P+step;vector<float> x=H;
        for(int li=0;li<NL;li++){TL&LY=Ls[li];vector<float> hn((size_t)L*D);for(int i=0;i<L;i++)rmsnorm(&x[(size_t)i*D],D,LY.iln.data(),&hn[(size_t)i*D]);
            auto Q=lin(hn.data(),L,D,LY.qw,nullptr,NH*HD),K=lin(hn.data(),L,D,LY.kw,nullptr,NKV*HD),V=lin(hn.data(),L,D,LY.vw,nullptr,NKV*HD);
            for(int i=0;i<L;i++){for(int h=0;h<NH;h++){float*q=&Q[(size_t)i*NH*HD+h*HD];float tp[128];rmsnorm(q,HD,LY.qn.data(),tp);for(int d=0;d<HD;d++)q[d]=tp[d];rope(q,i);}
                for(int h=0;h<NKV;h++){float*k=&K[(size_t)i*NKV*HD+h*HD];float tp[128];rmsnorm(k,HD,LY.kn.data(),tp);for(int d=0;d<HD;d++)k[d]=tp[d];rope(k,i);}}
            vector<float> ao((size_t)L*NH*HD,0.f);
            #pragma omp parallel for
            for(int i=0;i<L;i++)for(int h=0;h<NH;h++){int kvh=h/(NH/NKV);const float*qi=&Q[(size_t)i*NH*HD+h*HD];vector<float> sc(i+1);float mx=-1e30f;
                for(int j=0;j<=i;j++){const float*kj=&K[(size_t)j*NKV*HD+kvh*HD];double d=0;for(int t=0;t<HD;t++)d+=qi[t]*kj[t];float v=(float)d*scale;sc[j]=v;if(v>mx)mx=v;}
                double sm=0;for(int j=0;j<=i;j++){sc[j]=expf(sc[j]-mx);sm+=sc[j];}float iv=(float)(1.0/sm);float*aoi=&ao[(size_t)i*NH*HD+h*HD];
                for(int j=0;j<=i;j++){float w=sc[j]*iv;const float*vj=&V[(size_t)j*NKV*HD+kvh*HD];for(int t=0;t<HD;t++)aoi[t]+=w*vj[t];}}
            auto attn=lin(ao.data(),L,NH*HD,LY.ow,nullptr,D);for(size_t i=0;i<x.size();i++)x[i]+=attn[i];
            vector<float> h2((size_t)L*D);for(int i=0;i<L;i++)rmsnorm(&x[(size_t)i*D],D,LY.pln.data(),&h2[(size_t)i*D]);
            auto g=lin(h2.data(),L,D,LY.gate,nullptr,IM),u=lin(h2.data(),L,D,LY.up,nullptr,IM);for(size_t i=0;i<g.size();i++)g[i]=silu(g[i])*u[i];
            auto dn=lin(g.data(),L,IM,LY.down,nullptr,D);for(size_t i=0;i<x.size();i++)x[i]+=dn[i];}
        float hn[1024];rmsnorm(&x[(size_t)(L-1)*D],D,fnorm.data(),hn);long best=-1;double bv=-1e30;
        #pragma omp parallel
        {long lb=-1;double lv=-1e30;
         #pragma omp for nowait
         for(long v=0;v<VOCAB;v++){const float*e=&emb[(size_t)v*D];double s=0;for(int d=0;d<D;d++)s+=hn[d]*e[d];if(s>lv){lv=s;lb=v;}}
         #pragma omp critical
         {if(lv>bv){bv=lv;best=lb;}}}
        int tok=(int)best;out.push_back(tok);if(tok==151643||tok==151645)break;
        H.insert(H.end(),&emb[(size_t)tok*D],&emb[(size_t)tok*D]+D);}
    return out;
}

int main(int argc,char**argv){
    // load audio (wav.f32 for demo) and run full pipeline
    vector<float> wav;{FILE*f=fopen((ROOT+"/wav.f32").c_str(),"rb");float b;while(fread(&b,4,1,f)==1)wav.push_back(b);fclose(f);}
    int T=(int)((wav.size()+400)/160 - 0); // recompute below
    auto mel=mel_extract(wav); T=mel.size()/128; printf("mel (128,%d)\n",T);
    g_fe.load_param("./convert_asr_ncnn/frontend.ncnn.param"); g_fe.load_model("./convert_asr_ncnn/frontend.ncnn.bin");
    int N; auto ein=frontend(mel,T,N); printf("enc_in (%d,896)\n",N);
    auto ae=encoder(ein,N); printf("audio_embed (%d,1024)\n",N);
    // build prompt: template + N audio placeholders + language English
    const int D=1024;
    vector<int> pre={151644,8948,198,151645,198, 151644,872,198,151669};
    vector<int> post={151670,151645,198, 151644,77091,198, 11528,6364,151704};
    int P=pre.size()+N+post.size();
    auto emb=Tk("embed_tokens_weight",(size_t)151936*D);
    vector<float> ie; ie.reserve((size_t)P*D);
    auto push_emb=[&](int id){ie.insert(ie.end(),&emb[(size_t)id*D],&emb[(size_t)id*D]+D);};
    for(int t:pre)push_emb(t);
    for(int a=0;a<N;a++)ie.insert(ie.end(),&ae[(size_t)a*D],&ae[(size_t)a*D]+D);
    for(int t:post)push_emb(t);
    printf("prompt P=%d\n",P);
    auto toks=thinker(ie,P,emb);
    // BPE decode
    FILE*vf=fopen((TOK+"/vocab_bytes.bin").c_str(),"rb");unsigned cnt;size_t rr=fread(&cnt,4,1,vf);(void)rr;vector<string> vocab(cnt);
    for(unsigned i=0;i<cnt;i++){unsigned short len;if(fread(&len,2,1,vf)!=1)break;string s(len,0);if(len)rr=fread(&s[0],1,len,vf);vocab[i]=s;}fclose(vf);
    string text;for(int t:toks){if(t==151645||t==151643)break;if(t>=0&&t<(int)cnt)text+=vocab[t];}
    printf("\n=== TRANSCRIPTION ===\n%s\n",text.c_str());
    // compare golden
    FILE*tf=fopen((ROOT+"/text.txt").c_str(),"rb");string gt;{char b[4096];size_t n;while((n=fread(b,1,4096,tf))>0)gt.append(b,n);}fclose(tf);
    printf("\nvs golden: %s\n", text==gt?"== EXACT MATCH":"!= MISMATCH");
    return 0;
}
