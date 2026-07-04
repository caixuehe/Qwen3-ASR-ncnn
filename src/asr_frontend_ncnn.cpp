// P5: audio front-end via ncnn (conv2d x3 + conv_out run through ncnn::Net),
// C++ handles chunking + positional embedding + flatten. Verify enc_in vs golden.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include "net.h"
using namespace std;
static const int NMEL=128, WIN=100, D=896;
static const char* ENC="./convert_asr_enc";
static vector<float> rd(const string&n,size_t c){FILE*f=fopen((string(ENC)+"/"+n+".f32").c_str(),"rb");if(!f){fprintf(stderr,"open %s\n",n.c_str());exit(1);}vector<float>v(c);size_t r=fread(v.data(),4,c,f);fclose(f);(void)r;return v;}
static inline int convlen(int x){return (x-1)/2+1;}

int main(){
    // mel golden (128,T)
    int T=1484;
    vector<float> mel=rd("../convert_asr/mel",(size_t)NMEL*T);
    auto pe=rd("positional_embedding",(size_t)1500*D);
    ncnn::Net fe;
    if(fe.load_param("./convert_asr_ncnn/frontend.ncnn.param")||fe.load_model("./convert_asr_ncnn/frontend.ncnn.bin")){fprintf(stderr,"load ncnn fail\n");return 1;}

    vector<int> cl;int rem=T;while(rem>0){int c=rem>=WIN?WIN:rem;cl.push_back(c);rem-=c;}
    vector<float> enc_in;int nrow=0;
    for(size_t ci=0;ci<cl.size();ci++){
        ncnn::Mat in(WIN,NMEL,1); in.fill(0.f); float*p=in.channel(0);
        int base=(int)ci*WIN;
        for(int f=0;f<NMEL;f++)for(int t=0;t<WIN;t++){int tt=base+t; if(tt<T) p[f*WIN+t]=mel[(size_t)f*T+tt];}
        ncnn::Extractor ex=fe.create_extractor(); ex.input("in0",in);
        ncnn::Mat out; ex.extract("out0",out);   // (w=896,h=13)
        int valid=convlen(convlen(convlen(cl[ci])));
        for(int t=0;t<valid;t++){ const float*r=out.row(t);
            for(int d=0;d<D;d++) enc_in.push_back(r[d]+pe[(size_t)t*D+d]); nrow++; }
    }
    printf("enc_in rows=%d\n",nrow);
    auto g=rd("enc_in",(size_t)nrow*D);
    double md=0,ss=0,gg=0;for(size_t i=0;i<enc_in.size();i++){double d=enc_in[i]-g[i];if(fabs(d)>md)md=fabs(d);ss+=d*d;gg+=g[i]*g[i];}
    printf("enc_in (ncnn frontend) vs golden: maxdiff=%.3e relRMS=%.3e\n",md,sqrt(ss/gg));
    return 0;
}
