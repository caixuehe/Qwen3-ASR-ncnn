"""P4 prep: dump id->bytes table for BPE decode; show prompt template structure."""
import numpy as np, torch, os, struct
from qwen_asr import Qwen3ASRModel
OUT="./convert_asr_tok"; os.makedirs(OUT, exist_ok=True)
m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
tok = m.processor.tokenizer
print("tokenizer:", type(tok).__name__, "vocab", tok.vocab_size)

# byte-level decoder: token-string char -> byte
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
b2u = bytes_to_unicode(); u2b = {v:k for k,v in b2u.items()}

vocab_size = len(tok.get_vocab())
id2tok = {i:t for t,i in tok.get_vocab().items()}
print("full vocab size", vocab_size)

# dump per-id raw bytes, length-prefixed (uint16 len + bytes)
with open(f"{OUT}/vocab_bytes.bin","wb") as f:
    maxid = max(id2tok.keys())
    f.write(struct.pack("<I", maxid+1))
    for i in range(maxid+1):
        t = id2tok.get(i, None)
        if t is None: raw=b""
        else:
            try: raw = bytes(u2b[c] for c in t)
            except KeyError: raw = t.encode("utf-8")  # added/special tokens
        f.write(struct.pack("<H", len(raw))); f.write(raw)
print("dumped vocab_bytes.bin")

# gen tokens -> text (python sanity)
gen = np.fromfile("./convert_asr/gen_ids.i32", dtype=np.int32)
P=211; gentext_ids = gen[P:].tolist()
text = tok.decode(gentext_ids, skip_special_tokens=True)
print("decoded gen text:", repr(text))
open(f"{OUT}/gen_text_ids.txt","w").write(" ".join(map(str,gentext_ids)))

# show prompt template (decode prompt_ids specials)
pid = np.fromfile("./convert_asr/prompt_ids.i32", dtype=np.int32).tolist()
print("prompt len", len(pid))
# non-audio-placeholder tokens (skip 151676 runs)
comp=[]; run=0
for t in pid:
    if t==151676: run+=1
    else:
        if run: comp.append(f"<151676>x{run}"); run=0
        comp.append(f"{t}:{repr(id2tok.get(t,'?'))}")
if run: comp.append(f"<151676>x{run}")
print("prompt template:", " ".join(comp))
