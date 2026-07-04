import torch
from qwen_asr import Qwen3ASRModel
m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
model = m.model
print("top type:", type(model).__name__)
def walk(mod, prefix="", depth=0):
    for n, c in mod.named_children():
        print("  "*depth + f"{prefix}{n}: {type(c).__name__}")
        if depth < 1:
            walk(c, "", depth+1)
walk(model)
cfg = model.config
print("=== config top ===")
for k in dir(cfg):
    if k.startswith("_"): continue
    v = getattr(cfg, k, None)
    if isinstance(v, (int, float, str, bool)) or (hasattr(v,'__dict__')==False and not callable(v)):
        pass
# print sub-configs
for name in ["audio_config","text_config","thinker_config"]:
    sc = getattr(cfg, name, None)
    if sc is not None:
        print(f"--- {name} ---")
        for k in ["d_model","num_mel_bins","encoder_layers","downsample_hidden_size","output_dim","n_window","hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim","intermediate_size","vocab_size","rope_theta","rms_norm_eps"]:
            if hasattr(sc,k): print(f"   {k} = {getattr(sc,k)}")
