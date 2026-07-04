// P4: (A) build inputs_embeds from prompt_ids + audio_embed (inject at 151676 placeholders);
//     (B) byte-level BPE decode of generated tokens -> utf8 text. Both vs golden.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
using namespace std;
static const int D=1024; static const long VOCAB=151936;
static const int AUDIO_TOK=151676;

static vector<int> rdi(const string&p){ FILE*f=fopen(p.c_str(),"rb"); vector<int>v; int x; while(fread(&x,4,1,f)==1)v.push_back(x); fclose(f); return v; }

int main(){
    // ---------- Part A: inputs_embeds ----------
    auto pid = rdi("./convert_asr/prompt_ids.i32");
    int P=pid.size();
    FILE*ef=fopen("./convert_asr_thinker/embed_tokens_weight.f32","rb");
    FILE*af=fopen("./convert_asr/audio_embed.f32","rb");
    // audio_embed (193,1024)
    vector<float> ae; { float b; while(fread(&b,4,1,af)==1) ae.push_back(b);} fclose(af);
    int NA=ae.size()/D;
    vector<float> ie((size_t)P*D);
    int ai=0;
    for(int i=0;i<P;i++){
        if(pid[i]==AUDIO_TOK){ memcpy(&ie[(size_t)i*D], &ae[(size_t)ai*D], D*4); ai++; }
        else { long id=pid[i]; fseek(ef,(long)id*D*4,SEEK_SET); size_t r=fread(&ie[(size_t)i*D],4,D,ef); (void)r; }
    }
    fclose(ef);
    printf("Part A: P=%d audio injected=%d/%d\n",P,ai,NA);
    // compare golden
    FILE*gf=fopen("./convert_asr_thinker/inputs_embeds.f32","rb");
    vector<float> g((size_t)P*D); size_t rr=fread(g.data(),4,g.size(),gf); fclose(gf); (void)rr;
    double md=0,ss=0,gg=0; for(size_t i=0;i<ie.size();i++){double d=ie[i]-g[i];if(fabs(d)>md)md=fabs(d);ss+=d*d;gg+=g[i]*g[i];}
    printf("inputs_embeds vs golden: maxdiff=%.3e relRMS=%.3e\n",md,sqrt(ss/gg));

    // ---------- Part B: BPE decode ----------
    // load vocab_bytes: uint32 count, then per-id uint16 len + bytes
    FILE*vf=fopen("./convert_asr_tok/vocab_bytes.bin","rb");
    unsigned cnt; size_t r2=fread(&cnt,4,1,vf); (void)r2;
    vector<string> vocab(cnt);
    for(unsigned i=0;i<cnt;i++){ unsigned short len; if(fread(&len,2,1,vf)!=1)break; string s(len,0); if(len)r2=fread(&s[0],1,len,vf); vocab[i]=s; }
    fclose(vf);
    // gen text ids
    auto gen = rdi("./convert_asr/gen_ids.i32");
    string text;
    for(size_t i=P;i<gen.size();i++){ int t=gen[i]; if(t==151645||t==151643) break; if(t>=0&&t<(int)cnt) text+=vocab[t]; }
    // golden text
    FILE*tf=fopen("./convert_asr/text.txt","rb"); string gt; { char b[4096]; size_t n; while((n=fread(b,1,4096,tf))>0) gt.append(b,n);} fclose(tf);
    printf("Part B decoded: %s\n", text.c_str());
    printf("BPE decode %s golden (len %zu vs %zu)\n", text==gt?"== MATCH":"!= MISMATCH", text.size(), gt.size());
    return 0;
}
