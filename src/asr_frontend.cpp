// P2-A: Qwen3-ASR audio front-end (mel -> conv2d x3 -> conv_out -> +pos -> flatten -> enc_in).
// Chunk mel into 100-frame windows (pad tail to 100), each -> Conv2d(k3,s2,p1)x3 + gelu ->
// (480,16,13) -> conv_out Linear(7680->896) -> +sinusoid pos -> keep valid steps -> concat.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
using namespace std;
static const int NMEL=128, WIN=100, DSH=480, D=896;
static const char* DIR="./convert_asr_enc";
static vector<float> rd(const string&n,size_t cnt){
    FILE*f=fopen((string(DIR)+"/"+n+".f32").c_str(),"rb");
    if(!f){fprintf(stderr,"open %s\n",n.c_str());exit(1);}
    vector<float> v(cnt); size_t r=fread(v.data(),4,cnt,f); fclose(f);
    if(r!=cnt){fprintf(stderr,"read %s %zu/%zu\n",n.c_str(),r,cnt);exit(1);} return v;
}
static inline float gelu(float x){ return 0.5f*x*(1.0f+erff(x*0.70710678f)); }
static inline int convlen(int x){ return (x-1)/2+1; }

// conv2d k3 s2 p1: in[Cin,Hin,Win] -> out[Cout,Hout,Wout]
static vector<float> conv2d(const vector<float>&in,int Cin,int Hin,int Wi,
                            const vector<float>&W,const vector<float>&b,int Cout,int&Ho,int&Wo){
    Ho=convlen(Hin); Wo=convlen(Wi);
    vector<float> out((size_t)Cout*Ho*Wo);
    for(int oc=0;oc<Cout;oc++) for(int oh=0;oh<Ho;oh++) for(int ow=0;ow<Wo;ow++){
        double s=b[oc];
        for(int ic=0;ic<Cin;ic++){ const float*inp=&in[(size_t)ic*Hin*Wi];
            const float*w=&W[(((size_t)oc*Cin+ic)*3)*3];
            for(int kh=0;kh<3;kh++){ int ih=oh*2-1+kh; if(ih<0||ih>=Hin)continue;
                for(int kw=0;kw<3;kw++){ int iw=ow*2-1+kw; if(iw<0||iw>=Wi)continue;
                    s+=inp[ih*Wi+iw]*w[kh*3+kw]; } } }
        out[((size_t)oc*Ho+oh)*Wo+ow]=(float)s;
    }
    for(auto&z:out) z=gelu(z);
    return out;
}

int main(){
    // mel golden (128, T)
    int T=1484;
    vector<float> mel = rd("../convert_asr/mel",(size_t)NMEL*T); // stored (128,T) row-major
    // weights
    auto c1w=rd("conv2d1_weight",(size_t)DSH*1*9), c1b=rd("conv2d1_bias",DSH);
    auto c2w=rd("conv2d2_weight",(size_t)DSH*DSH*9), c2b=rd("conv2d2_bias",DSH);
    auto c3w=rd("conv2d3_weight",(size_t)DSH*DSH*9), c3b=rd("conv2d3_bias",DSH);
    int FDIM=16; // freq after 128->64->32->16
    auto cow=rd("conv_out_weight",(size_t)D*(DSH*FDIM));       // (896, 7680)
    auto pe =rd("positional_embedding",(size_t)1500*D);

    // chunk lengths: 14x100 + 84
    vector<int> clens; int rem=T; while(rem>0){ int c=rem>=WIN?WIN:rem; clens.push_back(c); rem-=c; }
    printf("chunks=%zu lens: ...%d\n", clens.size(), clens.back());

    vector<float> enc_in; enc_in.reserve((size_t)193*D);
    int nrow=0;
    for(size_t ci=0; ci<clens.size(); ci++){
        // build padded chunk (1,128,100): time window [ci*100, ci*100+100), zero-pad
        vector<float> chunk((size_t)NMEL*WIN, 0.f);
        int base=(int)ci*WIN;
        for(int mch=0;mch<NMEL;mch++) for(int tt=0; tt<WIN; tt++){
            int t=base+tt; if(t<T) chunk[(size_t)mch*WIN+tt]=mel[(size_t)mch*T+t];
        }
        int H1,W1,H2,W2,H3,W3;
        auto o1=conv2d(chunk,1,NMEL,WIN,c1w,c1b,DSH,H1,W1);
        auto o2=conv2d(o1,DSH,H1,W1,c2w,c2b,DSH,H2,W2);
        auto o3=conv2d(o2,DSH,H2,W2,c3w,c3b,DSH,H3,W3);   // (480,16,13)
        // conv_out: for each time t(0..W3-1): feat[c*16+f]=o3[c][f][t]; Linear(7680->896)
        int Tc=W3;
        int valid=convlen(convlen(convlen(clens[ci])));   // valid steps for this chunk
        for(int t=0;t<valid;t++){
            // build 7680 feature
            vector<float> feat((size_t)DSH*H3);
            for(int c=0;c<DSH;c++) for(int f=0;f<H3;f++) feat[c*H3+f]=o3[((size_t)c*H3+f)*W3+t];
            // Linear + pos
            for(int o=0;o<D;o++){ const float*w=&cow[(size_t)o*(DSH*H3)]; double s=0;
                for(int k=0;k<DSH*H3;k++) s+=feat[k]*w[k];
                enc_in.push_back((float)s + pe[(size_t)t*D+o]);
            }
            nrow++;
        }
    }
    printf("enc_in rows=%d (expect 193)\n", nrow);
    // compare
    auto g=rd("enc_in",(size_t)nrow*D);
    double md=0,ss=0,gg=0; for(size_t i=0;i<enc_in.size();i++){double d=enc_in[i]-g[i];if(fabs(d)>md)md=fabs(d);ss+=d*d;gg+=g[i]*g[i];}
    printf("enc_in vs golden: maxdiff=%.3e relRMS=%.3e\n",md,sqrt(ss/gg));
    return 0;
}
