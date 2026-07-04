"""P5: pnnx-convert the audio-encoder conv2d subsampling stack to ncnn (fp16=False!).
Verify ncnn output matches PyTorch on a sample (1,1,128,100) chunk."""
import numpy as np, torch, torch.nn as nn, os
from qwen_asr import Qwen3ASRModel
OUT="./convert_asr_ncnn"; os.makedirs(OUT, exist_ok=True)
m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
at = m.model.thinker.audio_tower

class ConvStack(nn.Module):
    def __init__(self, at):
        super().__init__()
        self.c1, self.c2, self.c3 = at.conv2d1, at.conv2d2, at.conv2d3
    def forward(self, x):
        import torch.nn.functional as F
        x = F.gelu(self.c1(x)); x = F.gelu(self.c2(x)); x = F.gelu(self.c3(x))
        return x

net = ConvStack(at).eval()
x = torch.randn(1,1,128,100)
with torch.no_grad(): y_ref = net(x)
print("ref out", y_ref.shape)

import pnnx
os.chdir(OUT)
np.save("../_x.npy", x.numpy())
mod = pnnx.export(net, "conv_stack.pt", x, fp16=False)
os.chdir("..")
print("pnnx exported (fp16=False)")

# verify with python ncnn
import ncnn
n = ncnn.Net()
n.load_param(f"{OUT}/conv_stack.ncnn.param")
n.load_model(f"{OUT}/conv_stack.ncnn.bin")
ex = n.create_extractor()
ex.input("in0", ncnn.Mat(x.numpy()[0]))  # (1,128,100)
_, out = ex.extract("out0")
yn = np.array(out)  # (480,16,13)
yr = y_ref.numpy()[0]
print("ncnn out", yn.shape, "ref", yr.shape)
d = np.abs(yn - yr)
print(f"conv ncnn vs torch: maxdiff={d.max():.3e} relRMS={np.sqrt((d**2).sum()/(yr**2).sum()):.3e}")
# check bin size (fp16 would be half)
print("bin bytes:", os.path.getsize(f"{OUT}/conv_stack.ncnn.bin"))
