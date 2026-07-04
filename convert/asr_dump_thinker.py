"""P3 prep: dump Thinker LLM weights + inputs_embeds golden (audio injected)."""
import numpy as np, torch, librosa, functools, os, json
from qwen_asr import Qwen3ASRModel
OUT="./convert_asr_thinker"; os.makedirs(OUT, exist_ok=True)
m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
tm = m.model.thinker.model   # Qwen3ASRThinkerTextModel
def save(name,t): np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tofile(f"{OUT}/{name}.f32")

manifest={}
for k,v in tm.named_parameters():
    fn=k.replace(".","_"); save(fn,v); manifest[k]=list(v.shape)
json.dump(manifest, open(f"{OUT}/manifest.json","w"), indent=1)
print("thinker weights:", len(manifest), "embed", manifest.get("embed_tokens.weight"))

# hook to capture inputs_embeds fed to the text model (after audio injection)
cap={}
orig=tm.forward
@functools.wraps(orig)
def hook(*a, **k):
    ie=k.get("inputs_embeds", None)
    if ie is not None and "ie" not in cap:
        cap["ie"]=ie.detach().cpu().numpy().astype(np.float32)
    return orig(*a,**k)
tm.forward=hook

m.transcribe(audio=librosa.example("libri1"), language="English")
ie=cap["ie"][0] if cap["ie"].ndim==3 else cap["ie"]
np.ascontiguousarray(ie).tofile(f"{OUT}/inputs_embeds.f32")
print("inputs_embeds", ie.shape)
