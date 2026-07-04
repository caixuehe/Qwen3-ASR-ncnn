"""P1 prep: dump 16kHz waveform + Whisper mel filterbank; verify feature extractor
reproduces golden mel. C++ will do STFT + matmul(filters) + log-norm."""
import numpy as np, torch, librosa, os
from qwen_asr import Qwen3ASRModel
OUT = "./convert_asr"; os.makedirs(OUT, exist_ok=True)

m = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B", dtype=torch.float32, attn_implementation="eager")
fe = m.processor.feature_extractor
print("feature_extractor:", type(fe).__name__)
print("n_fft", fe.n_fft, "hop", fe.hop_length, "n_mels", fe.feature_size, "sr", fe.sampling_rate)
mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)   # (num_freq=201, n_mels=128) or (128,201)
print("mel_filters", mel_filters.shape)

# load audio at 16kHz mono (Whisper standard)
wav, sr = librosa.load(librosa.example("libri1"), sr=16000, mono=True)
wav = wav.astype(np.float32)
print("wav", wav.shape, "sr", sr)

# run the real feature extractor -> mel, compare to golden
feat = fe(wav, sampling_rate=16000, return_tensors="np")
mel = np.asarray(feat["input_features"][0], dtype=np.float32)
print("fe mel", mel.shape)
golden = np.fromfile(f"{OUT}/mel.f32", dtype=np.float32)
gm = golden.reshape(128, -1)
n = min(mel.shape[1], gm.shape[1])
d = np.abs(mel[:, :n] - gm[:, :n])
print(f"fe mel vs golden: shapes {mel.shape} vs {gm.shape}, maxdiff(overlap)={d.max():.3e}")

# dump for C++
wav.tofile(f"{OUT}/wav.f32")
np.ascontiguousarray(mel_filters).tofile(f"{OUT}/mel_filters.f32")
with open(f"{OUT}/mel_cfg.txt","w") as f:
    f.write(f"{fe.n_fft} {fe.hop_length} {fe.feature_size} {mel_filters.shape[0]} {mel_filters.shape[1]} {len(wav)}\n")
print("dumped wav, mel_filters, mel_cfg")
