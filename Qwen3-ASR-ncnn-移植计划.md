# Qwen3-ASR ncnn 移植计划

> 犀牛鸟 Issue #6790（Tencent/ncnn）· 难度：高
> 目标：C++/ncnn 复现 Qwen3-ASR（0.6B），音频→文本，输出与 PyTorch 一致，CMake 双平台，发 Discussion + repo。

## 架构（读源码 `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`）

Qwen-Omni "Thinker" 结构，音频-LLM：

```
音频波形 → [mel 特征] → [Audio Encoder] → 音频 embedding → [Thinker LLM 自回归] → 文本 token → BPE 解码 → 文字
```

- **Audio Encoder**（`Qwen3ASRAudioEncoder`）：
  mel(num_mel_bins) → Conv2d×3(stride2,下采样×8) + conv_out(Linear) → +SinusoidsPositionEmbedding
  → N 层 `Qwen3ASRAudioEncoderLayer`（`Qwen3ASRAudioAttention`，**块内双向注意力**，非因果）
  → ln_post → proj1→act→proj2 → 音频 embedding(output_dim)。分块处理(n_window/chunk)。
- **Thinker**（`Qwen3ASRThinkerTextModel`）：Qwen 风格自回归 LLM（RoPE + QK-norm + GQA + SwiGLU + RMSNorm），
  吃 [音频 embedding + prompt] → greedy 生成文本 token。
- 文本 tokenizer：Qwen BPE（解码方向：token id → 文字）。

## 与 TTS 项目的对应（大量复用）

| ASR 组件 | 对应 TTS | 复用 |
|---|---|---|
| Thinker 自回归 LLM | Talker | ≈ 照搬（C++ attention + ncnn 前馈 + KV-cache + greedy） |
| Audio Encoder 注意力层 | codec transformer | 复用分工（但**双向**、有 conv 下采样） |
| Conv2d 下采样 | codec 卷积栈 | pnnx→ncnn，`fp16=False` |
| BPE 解码 | BPE 编码(`bpe_tokenizer.cpp`) | 反向 |
| CMake/CI/权重dump框架 | 全套 | 直接搬 |

**直接复用的教训**：ncnn 无 batch 维→前馈 ncnn/注意力 C++；pnnx `fp16=False`；transformers vmap mask 转换期换普通张量；greedy 才能对齐；单算子隔离测不可靠→全栈 relRMS；KV-cache 加速。

## 新增（ASR 特有，要重点搞）
1. **mel 特征提取**：STFT + mel 滤波器 → C++（信号处理，新）。
2. **Conv2d×3 下采样**：2D 卷积（TTS 是 1D），pnnx→ncnn。
3. **双向 chunked 编码器注意力**：非因果，块内双向。
4. **BPE 解码方向**：token id → bytes → utf8 文字。

## 分阶段（TDD，每步对 golden）

- [x] **P0 黄金参考** ✅：greedy transcribe，dump mel(128,1484)/audio_embed(193,1024)/prompt(211)/gen(263)/text；确定性 True。样例=librosa libri1。
- [x] **P1 mel 特征（C++）** ✅：`src/mel_features.cpp`，reflect-pad + hann + STFT(DFT,n_fft400/hop160) + mel滤波(201×128) + log10全局归一。vs golden **maxdiff 1.9e-5, relRMS 1.3e-6**。配方=`fe(wav,padding=True,truncation=False)`。
- [x] **P2 Audio Encoder（C++ 正确性）** ✅：
  - P2-A 前端 `src/asr_frontend.cpp`：mel 切 15 块(14×100+84)→Conv2d(k3s2p1)×3+gelu→(480,16,13)→conv_out(7680→896)→+正弦位置→按 valid 展平(13×14+11=193)。vs enc_in golden **maxdiff 4e-6**。
  - P2-B transformer `src/asr_encoder.cpp`：18层 pre-LN(LayerNorm+MHA14头/hd64/bias+gelu MLP)+ln_post+proj1/gelu/proj2。**关键：eager+mask=None → 全局双向注意力**(cu_seqlens 块对角只是 FA2 varlen 路径，非 eager)。vs audio_embed golden **maxdiff 3e-7**。
  - 待交付：conv/MLP/proj 前馈转 ncnn(pnnx，fp16=False)。
- [x] **P3 Thinker LLM（C++, recompute, greedy）** ✅：`src/asr_thinker.cpp`，28层 Qwen3(RMSNorm+QK-norm+RoPE θ1e6 hd128+GQA16/8 causal+SwiGLU)，tie lm_head=embed_tokens 反投影 argmax，eos∈{151643,151645}。从 inputs_embeds golden 起 **生成 52/52 token 全对**（末尾 eos 151645）。mRoPE[24,20,20] 文本下退化普通 RoPE，直接搬 Talker。待优化：KV-cache。
- [x] **P4 注入 + BPE 解码** ✅：`src/asr_p4.cpp`。
  - Part A：prompt_ids + audio_embed → inputs_embeds（151676 占位替换成 audio_embed 行），vs golden **maxdiff 0.0**（精确）。
  - Part B：byte-level BPE 解码(id→字节表 `convert_asr_tok/vocab_bytes.bin`，拼字节→utf8)，vs golden 文本 **完全一致**。
  - prompt 模板(211)：`<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>[193×151676]<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage ĠEnglish<asr_text>`；生成从 151704(`<asr_text>`)后开始。
  - 待整合：单可执行 wav→文字(串起 P1-P4，纯 glue)。
- [~] **P5 交付（进行中）**：
  - [x] **单可执行 `src/qwen3_asr.cpp`** ✅：wav→mel→frontend→encoder→自建prompt(211)+注入→Thinker(greedy)→BPE。端到端 **文本 == PyTorch 逐字一致（EXACT MATCH）**。纯 C++/OpenMP。
  - [x] **pnnx→ncnn 路径打通** ✅：conv2d 下采样栈转 ncnn(`convert/asr_convert_conv.py`→`convert_asr_ncnn/conv_stack.ncnn.*`)，vs torch **maxdiff 2.1e-6**，`fp16=False`(bin 16.6MB 满 fp32)，pnnx 各 pass 干净(conv2d 全支持，无 vmap 坑)。encoder/Thinker MLP+proj+lm_head 同模式待批量转。
  - [ ] 用 ncnn::Net 接回 C++ 可执行(需 ncnn build + CMake)。
  - [x] **CMake + CI + README + .gitignore** ✅：`CMakeLists.txt`(MSVC /utf-8 /O2 + OpenMP，构建 qwen3_asr + 5 组件测试，school 上验证通过)、`.github/workflows/build.yml`(ubuntu+windows matrix，纯 C++ 无外部依赖 CI 简单)、README(流水线+结果表+架构分工)、.gitignore。school git 已提交(无二进制)。
  - [x] **KV-cache Thinker** ✅：`src/asr_thinker_kv.cpp`，缓存每层 K(post-qknorm+rope)/V，prefill P 行后每步只算新 token 对 cache 注意力，O(T²)→O(T)。**match 52/52**。注：短序列下 lm_head(152k vocab argmax)是主开销，不受 KV 影响。待：整合进 qwen3_asr.cpp 主可执行。
  - [ ] 批量转 ncnn(encoder/Thinker MLP+proj+lm_head) + ncnn::Net 接回 C++。
  - [ ] 公开 repo(clone 到本地 push，去掉计划 md，OWNER→caixuehe) + Discussion(Tencent/ncnn Show and tell) + issue #6790 认领。

## 确认规格（读 config.json）
- **模型结构**：`model.thinker.{audio_tower(Qwen3ASRAudioEncoder), model(Qwen3ASRThinkerTextModel), lm_head}`。
- **mel**：WhisperFeatureExtractor，128 mel，n_fft=400，hop=160，16kHz，chunk 30s。→ **标准 Whisper mel，C++ 有成熟参考**。
- **Audio Encoder**：d_model=896，18层，14头，ffn3584，Conv2d×3(stride2,downsample_hidden=480)，gelu，输出 output_dim=1024，n_window=50/infer=800，conv_chunksize=500，max_src_pos=1500，正弦位置，scale_embedding=false。
- **Thinker LLM**：hidden1024，**28层，GQA16/8，head_dim128，IM3072，θ1e6，eps1e-6，silu，vocab151936，tie_word_embeddings=True**（≈TTS Talker config，仅 vocab 与 tie 不同，**C++ 大幅复用 Talker**；待确认 RoPE 是否 mRoPE、有无 QK-norm）。
- **特殊 token**：audio_start=151669，audio_end=151670，audio_token=151676(占位)，eos=[151643,151645]，pad=151643。**do_sample=False**（greedy，P0 已验证确定性）。
- **P0 黄金文本**（librosa libri1 样例）：`With her white paint and her scarlet smokestack, the Inverishield—...—between Inverishield and Cryonon.`；两次运行一致✅。

## 环境
- 复用 conda env `qwen-tts-ncnn`（已装 qwen-asr 0.0.6，transformers 4.57.6，pnnx，ncnn）。
- school `/data/bowen/cxhe/Qwen3-ASR-ncnn/`；空闲 GPU 6/7；长任务 tmux。
- 模型 `models/Qwen3-ASR-0.6B`（1.88GB）。
