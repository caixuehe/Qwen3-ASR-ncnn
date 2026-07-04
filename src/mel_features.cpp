// P1: Whisper mel feature extraction in C++ (Qwen3-ASR front-end).
// Reproduces WhisperFeatureExtractor(padding=True, truncation=False): reflect-pad,
// hann-windowed STFT (n_fft=400, hop=160), power spectrum, mel filterbank matmul,
// log10 + global normalize (log+4)/4 after clamping to gmax-8.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
using namespace std;

static vector<float> readf(const string& p, size_t n){
    FILE* f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(1);}
    vector<float> v(n); size_t r=fread(v.data(),4,n,f); fclose(f);
    if(r!=n){fprintf(stderr,"read %s: %zu/%zu\n",p.c_str(),r,n);exit(1);} return v;
}

int main(int argc,char**argv){
    const char* dir = argc>1?argv[1]:"./convert_asr";
    const int NFFT=400, HOP=160, NMEL=128, NFREQ=201; // NFREQ=NFFT/2+1
    // read config for wav length + verify
    int nfft,hop,nmel,mf0,mf1; long wavlen;
    { char p[512]; snprintf(p,512,"%s/mel_cfg.txt",dir); FILE*f=fopen(p,"r");
      fscanf(f,"%d %d %d %d %d %ld",&nfft,&hop,&nmel,&mf0,&mf1,&wavlen); fclose(f); }
    printf("cfg n_fft=%d hop=%d n_mels=%d filters=(%d,%d) wavlen=%ld\n",nfft,hop,nmel,mf0,mf1,wavlen);

    vector<float> wav = readf(string(dir)+"/wav.f32", wavlen);
    vector<float> filt = readf(string(dir)+"/mel_filters.f32", (size_t)NFREQ*NMEL); // (201,128) row-major

    // reflect pad by NFFT/2 on both sides
    int pad=NFFT/2; long PL=wavlen+2*pad;
    vector<float> x(PL);
    for(long i=0;i<PL;i++){
        long j=i-pad;                    // index into wav
        if(j<0) j=-j;                    // reflect (no edge repeat)
        else if(j>=wavlen) j=2*wavlen-2-j;
        x[i]=wav[j];
    }
    long T = 1 + (PL-NFFT)/HOP; T -= 1;   // whisper drops last stft frame
    printf("frames T=%ld\n", T);

    // hann window (periodic) + DFT cos/sin tables
    vector<float> win(NFFT);
    for(int n=0;n<NFFT;n++) win[n]=0.5f-0.5f*cosf(2*M_PI*n/NFFT);
    vector<float> C((size_t)NFREQ*NFFT), S((size_t)NFREQ*NFFT);
    for(int k=0;k<NFREQ;k++) for(int n=0;n<NFFT;n++){
        double a=-2*M_PI*k*n/NFFT; C[(size_t)k*NFFT+n]=(float)cos(a); S[(size_t)k*NFFT+n]=(float)sin(a);
    }

    // mel[m][t]
    vector<float> mel((size_t)NMEL*T);
    vector<float> frame(NFFT), power(NFREQ);
    for(long t=0;t<T;t++){
        long off=t*HOP;
        for(int n=0;n<NFFT;n++) frame[n]=x[off+n]*win[n];
        for(int k=0;k<NFREQ;k++){
            const float*ck=&C[(size_t)k*NFFT]; const float*sk=&S[(size_t)k*NFFT];
            double re=0,im=0;
            for(int n=0;n<NFFT;n++){ re+=frame[n]*ck[n]; im+=frame[n]*sk[n]; }
            power[k]=(float)(re*re+im*im);
        }
        for(int m=0;m<NMEL;m++){
            double s=0; for(int k=0;k<NFREQ;k++) s+=(double)filt[(size_t)k*NMEL+m]*power[k];
            double lv=log10(s<1e-10?1e-10:s);
            mel[(size_t)m*T+t]=(float)lv;
        }
    }
    // global normalize
    float gmax=-1e30f;
    for(size_t i=0;i<mel.size();i++) if(mel[i]>gmax) gmax=mel[i];
    float floor=gmax-8.0f;
    for(size_t i=0;i<mel.size();i++){ float v=mel[i]<floor?floor:mel[i]; mel[i]=(v+4.0f)/4.0f; }

    // compare to golden
    vector<float> g = readf(string(dir)+"/mel.f32",(size_t)NMEL*T);
    double md=0,ss=0,gg=0; for(size_t i=0;i<mel.size();i++){ double d=mel[i]-g[i]; if(fabs(d)>md)md=fabs(d); ss+=d*d; gg+=g[i]*g[i]; }
    printf("mel (%d,%ld) vs golden: maxdiff=%.3e  relRMS=%.3e\n",NMEL,T,md,sqrt(ss/gg));
    // dump for later stages
    { FILE*f=fopen((string(dir)+"/mel_cpp.f32").c_str(),"wb"); fwrite(mel.data(),4,mel.size(),f); fclose(f); }
    return 0;
}
