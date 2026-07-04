"""P0: Qwen3-ASR golden reference. Transcribe a sample audio (greedy, float32, eager),
print the golden text, verify determinism."""
import logging, torch, librosa
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from qwen_asr import Qwen3ASRModel

audio = librosa.example("libri1")   # LibriSpeech English sample (cached)
print("audio:", audio)

model = Qwen3ASRModel.from_pretrained("./models/Qwen3-ASR-0.6B",
                                      dtype=torch.float32, attn_implementation="eager")
res = model.transcribe(audio=audio, language="English")
print("=== transcription ===")
print(repr(res))
t1 = res[0].text if hasattr(res[0], "text") else str(res[0])
res2 = model.transcribe(audio=audio, language="English")
t2 = res2[0].text if hasattr(res2[0], "text") else str(res2[0])
print("text:", t1)
print("DETERMINISM (two runs identical):", t1 == t2)
