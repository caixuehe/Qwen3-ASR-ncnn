"""P5: pnnx-convert full conv front-end (conv2d x3 + permute/view + conv_out) to ncnn.
Per-chunk: (1,1,128,100) -> (13, 896). fp16=False. Verify vs PyTorch."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, os
from qwen_asr import Qwen3ASRModel
OUT="./convert_asr_ncnn"; os.makedirs(OUT, exist_ok=True)
m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
at = m.model.thinker.audio_tower

class Frontend(nn.Module):
    def __init__(self, at):
        super().__init__()
        self.c1,self.c2,self.c3,self.conv_out = at.conv2d1,at.conv2d2,at.conv2d3,at.conv_out
    def forward(self, x):
        x=F.gelu(self.c1(x)); x=F.gelu(self.c2(x)); x=F.gelu(self.c3(x))
        b,c,f,t = x.size()
        x = x.permute(0,3,1,2).contiguous().view(b,t,c*f)
        return self.conv_out(x)   # (1,13,896)

net=Frontend(at).eval()
x=torch.randn(1,1,128,100)
with torch.no_grad(): yr=net(x)
print("ref", yr.shape)
import pnnx
os.chdir(OUT)
mod=pnnx.export(net,"frontend.pt",x,fp16=False)
os.chdir("..")
import ncnn
n=ncnn.Net(); n.load_param(f"{OUT}/frontend.ncnn.param"); n.load_model(f"{OUT}/frontend.ncnn.bin")
ex=n.create_extractor(); ex.input("in0", ncnn.Mat(x.numpy()[0]))
_,o=ex.extract("out0"); yn=np.array(o)
print("ncnn out", yn.shape, "ref", yr.numpy()[0].shape)
yrr=yr.numpy()[0]
# shapes may be (13,896) vs (896,13)-ish; align by size
yn2=yn.reshape(yrr.shape) if yn.size==yrr.size else yn
d=np.abs(yn2-yrr); print(f"frontend ncnn vs torch: maxdiff={d.max():.3e} relRMS={np.sqrt((d**2).sum()/(yrr**2).sum()):.3e}")
print("bin bytes:", os.path.getsize(f"{OUT}/frontend.ncnn.bin"))
