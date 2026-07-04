"""P0 golden dump: capture mel features, audio-encoder output, prompt/generated token ids."""
import numpy as np, torch, librosa, functools, os
from qwen_asr import Qwen3ASRModel
OUT = "./convert_asr"; os.makedirs(OUT, exist_ok=True)

m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
model = m.model
cap = {}

# hook audio encoder: capture input_features (mel) + output embeddings
at = model.thinker.audio_tower
orig_at = at.forward
@functools.wraps(orig_at)
def h_at(*a, **k):
    out = orig_at(*a, **k)
    feats = k.get("input_features", a[0] if a else None)
    if feats is not None and "mel" not in cap:
        cap["mel"] = feats.detach().cpu().numpy().astype(np.float32)
        # out may be a tensor or object
        oo = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        cap["audio_embed"] = (oo[0] if isinstance(oo, tuple) else oo).detach().cpu().numpy().astype(np.float32)
    return out
at.forward = h_at

# hook generate: capture input_ids (prompt) + generated ids
orig_gen = model.generate
@functools.wraps(orig_gen)
def h_gen(*a, **k):
    ids = k.get("input_ids", a[0] if a else None)
    if ids is not None and "prompt_ids" not in cap:
        cap["prompt_ids"] = ids.detach().cpu().numpy().astype(np.int64)
    out = orig_gen(*a, **k)
    seq = out.sequences if hasattr(out, "sequences") else out
    cap["gen_ids"] = seq.detach().cpu().numpy().astype(np.int64)
    return out
model.generate = h_gen

audio = librosa.example("libri1")
res = m.transcribe(audio=audio, language="English")
text = res[0].text
print("text:", text)

mel = cap["mel"]; ae = cap["audio_embed"]
print("mel", mel.shape, "audio_embed", ae.shape)
print("prompt_ids", cap["prompt_ids"].shape, "gen_ids", cap["gen_ids"].shape)
mel.tofile(f"{OUT}/mel.f32")
np.ascontiguousarray(ae).tofile(f"{OUT}/audio_embed.f32")
cap["prompt_ids"].reshape(-1).astype(np.int32).tofile(f"{OUT}/prompt_ids.i32")
cap["gen_ids"].reshape(-1).astype(np.int32).tofile(f"{OUT}/gen_ids.i32")
open(f"{OUT}/text.txt","w").write(text)
with open(f"{OUT}/asr_dims.txt","w") as f:
    f.write(f"mel {list(mel.shape)}\naudio_embed {list(ae.shape)}\nprompt {cap['prompt_ids'].shape[-1]}\ngen {cap['gen_ids'].shape[-1]}\n")
print("dumped to", OUT)
# also print the prompt id sequence around audio placeholders
pid = cap["prompt_ids"].reshape(-1)
print("prompt head:", pid[:12].tolist(), "...tail:", pid[-8:].tolist())
print("gen head:", cap["gen_ids"].reshape(-1)[:12].tolist())
