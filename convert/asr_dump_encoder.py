"""P2 prep: dump audio encoder weights + intermediate goldens for TDD."""
import numpy as np, torch, librosa, functools, os, json
from qwen_asr import Qwen3ASRModel
OUT="./convert_asr_enc"; os.makedirs(OUT, exist_ok=True)
m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
at = m.model.thinker.audio_tower

def save(name, t):
    np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tofile(f"{OUT}/{name}.f32")

# weights
sd = dict(at.named_parameters())
manifest={}
for k,v in sd.items():
    fn=k.replace(".","_")
    save(fn, v); manifest[k]=list(v.shape)
# positional embedding buffer
pe = at.positional_embedding.positional_embedding
save("positional_embedding", pe); manifest["positional_embedding"]=list(pe.shape)
json.dump(manifest, open(f"{OUT}/weights_manifest.json","w"), indent=1)
print("weights dumped:", len(manifest))

# hook intermediate goldens
cap={}
layers=at.layers
orig_fwd = at.forward
# capture hidden_states before layer loop + cu_seqlens by monkeypatching layer 0 input
h0 = layers[0].forward
@functools.wraps(h0)
def hook0(hidden_states, cu_seqlens, *a, **k):
    if "enc_in" not in cap:
        cap["enc_in"]=hidden_states.detach().cpu().numpy().astype(np.float32)
        cap["cu_seqlens"]=cu_seqlens.detach().cpu().numpy().astype(np.int64)
    out = h0(hidden_states, cu_seqlens, *a, **k)
    cap["layer0_out"]=out[0].detach().cpu().numpy().astype(np.float32)
    return out
layers[0].forward = hook0

audio = librosa.example("libri1")
m.transcribe(audio=audio, language="English")

for kk in ["enc_in","layer0_out"]:
    np.ascontiguousarray(cap[kk]).tofile(f"{OUT}/{kk}.f32")
cap["cu_seqlens"].astype(np.int32).tofile(f"{OUT}/cu_seqlens.i32")
print("enc_in", cap["enc_in"].shape, "layer0_out", cap["layer0_out"].shape, "cu_seqlens", cap["cu_seqlens"].tolist())
# final audio_embed already dumped in convert_asr/audio_embed.f32
print("done")
