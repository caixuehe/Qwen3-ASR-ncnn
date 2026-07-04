# Qwen3-ASR-ncnn

![build](https://github.com/OWNER/Qwen3-ASR-ncnn/actions/workflows/build.yml/badge.svg)

A from-scratch C++ port of [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (0.6B),
converting audio to text. Feed-forward stages convert to
[ncnn](https://github.com/Tencent/ncnn) via `pnnx`; attention and the autoregressive
loop are hand-written in C++. Every stage is checked numerically against the PyTorch
reference, and the end-to-end text is **bit-identical** to `qwen-asr`.

> 2026 腾讯犀牛鸟开源人才培养 · Issue #6790

## Pipeline

```
audio ──[Whisper mel: 128 bins, n_fft 400, hop 160]──▶ mel (128, T)
      ──[Conv2d ×3 stride-2 + gelu, per 100-frame chunk]──▶ subsampled
      ──[conv_out 7680→896 + sinusoidal pos]──▶ enc_in (N, 896)
      ──[18-layer audio encoder: LayerNorm + MHA(14h) + gelu MLP]──▶ audio_embed (N, 1024)
      ──[inject at 193 audio placeholders in the chat prompt]──▶ inputs_embeds (P, 1024)
      ──[28-layer Thinker LLM: RMSNorm + QK-norm + RoPE + GQA 16/8 + SwiGLU, greedy]──▶ token ids
      ──[byte-level BPE decode]──▶ text
```

One executable, `qwen3_asr`. Greedy decoding, deterministic.

## Results (vs PyTorch reference, `qwen-asr` 0.0.6, Qwen3-ASR-0.6B)

Sample: `librosa.example("libri1")` (LibriSpeech English).

| Stage | Metric |
|---|---|
| Whisper mel | maxdiff 1.9e-5 |
| Conv2d front-end → enc_in | maxdiff 4e-6 |
| Audio encoder → audio_embed | maxdiff 3e-7 |
| Audio injection → inputs_embeds | maxdiff 0.0 (exact) |
| Thinker LLM (generated tokens) | **52/52 identical** |
| BPE decode | exact |
| **End-to-end audio → text** | **bit-identical to PyTorch** |
| Conv stack (pnnx → ncnn, fp16=False) | maxdiff 2.1e-6 |

## Architecture split (why part ncnn, part C++)

ncnn has no batch dimension, so pnnx-converting a HuggingFace transformer whole-hog
produces `broadcast across batch axis` / `unbind along batch axis` ops that crash at
runtime. Each network is therefore split:

- **Conv2d subsampling, MLP, projections, lm_head** → converted to ncnn via `pnnx`.
- **Attention (encoder bidirectional + Thinker causal GQA), RoPE, autoregressive loop**
  → hand-written C++ around plain matmul / softmax.

Two lessons carried over from the sibling
[Qwen3-TTS-ncnn](https://github.com/OWNER/Qwen3-TTS-ncnn) port:

- `pnnx.export(..., fp16=False)` is mandatory — the default fp16 weight storage gives
  ~1e-3 relative error per layer, which deep stacks amplify.
- The audio encoder uses **full bidirectional attention** under eager attention; the
  `cu_seqlens` block-diagonal layout is only the FlashAttention-2 varlen path.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires CMake ≥ 3.16 and a C++17 compiler. OpenMP is used if available.

## Prepare weights

Weights are not checked in. Run the conversion once (needs the `qwen-asr` Python
package + `pnnx`, see `convert/`):

```bash
python convert/asr_golden.py         # golden mel / audio_embed / tokens / text
python convert/asr_p1prep.py         # 16kHz waveform + Whisper mel filterbank
python convert/asr_dump_encoder.py   # audio encoder weights + intermediate goldens
python convert/asr_dump_thinker.py   # Thinker LLM weights + inputs_embeds golden
python convert/asr_p4prep.py         # byte-level BPE id→bytes table
python convert/asr_convert_conv.py   # pnnx-export the conv2d stack to ncnn (fp16=False!)
```

## Run

```bash
OMP_NUM_THREADS=32 ./build/qwen3_asr
```

## Layout

```
src/            C++: qwen3_asr (end-to-end) + per-stage test harnesses
convert/        Python: weight dumps + pnnx conversion
```

## Credits

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — the model.
- [Tencent/ncnn](https://github.com/Tencent/ncnn) + `pnnx`.
- [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm) — reference for C++ LLM decoding on ncnn.
