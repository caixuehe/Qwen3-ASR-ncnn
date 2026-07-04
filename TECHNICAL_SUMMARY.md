# Qwen3-ASR 的 ncnn 移植：纯 C++ 端到端复现

在完成 [Qwen3-TTS 的移植](https://github.com/caixuehe/Qwen3-TTS-ncnn)之后，本项目进一步将 Qwen3-ASR（0.6B）移植至 ncnn。两者方向恰好相反：前者为文本到语音，后者为语音到文本。最终交付形式相同——一个 C++ 可执行文件，输入一段音频，输出转写文本，推理阶段仅依赖 ncnn，不再需要 PyTorch。

得益于 TTS 移植积累的基础，本次工作进展显著加快——如后文所述，ASR 中的自回归语言模型与 TTS 中的 Talker 结构高度一致，代码大部分可直接复用；真正需要从零实现的仅有音频前端。各环节的数值对齐结果如下：

| 环节 | 与 PyTorch 的差距 |
|---|---|
| Whisper mel 特征 | maxdiff 1.9e-5 |
| Conv2d 前端 → 编码器输入 | maxdiff 4e-6 |
| 音频编码器 → 音频 embedding | maxdiff 3e-7 |
| 音频注入 prompt → 输入 embedding | maxdiff 0（精确） |
| Thinker LLM 生成的 token | **52/52 完全一致** |
| BPE 解码 | 精确 |
| **端到端 音频 → 文字** | **与 PyTorch 逐字一致** |
| Conv 前端（pnnx→ncnn，fp16=False） | maxdiff 1.3e-5 |

完整代码见仓库：https://github.com/caixuehe/Qwen3-ASR-ncnn 。样例音频采用 `librosa.example("libri1")`（一段 LibriSpeech 英文语音）。

## 移植对象

Qwen3-ASR 采用 Qwen-Omni 系列的 "Thinker" 架构，属于音频-LLM。其整体推理链路如下：

```
音频 →[Whisper mel]→ mel →[音频编码器]→ 音频 embedding →[塞进对话 prompt 的占位符]→ 输入 embedding
     →[Thinker：28 层自回归 LLM，greedy]→ 文本 token →[byte-level BPE 解码]→ 文字
```

链路前半段为新增内容：Whisper 风格的 mel 特征提取，以及由卷积下采样与 18 层 Transformer 构成的音频编码器。后半段的 Thinker 为标准的 Qwen3 自回归解码器——接收音频 embedding 与提示词，逐 token 生成转写结果。

## 沿用自 TTS 的核心判断：ncnn 不建模 batch 维

这一结论在 TTS 项目中已得到充分验证：ncnn 不建模 batch 维，而 HuggingFace 实现中的注意力及相关算子大量依赖跨 batch 的广播与 unbind 操作，若经 pnnx 整体转换，在 ncnn 侧无法正常运行。因此本次沿用相同的分工策略：卷积、MLP、投影等前馈算子交由 pnnx 转换至 ncnn；注意力与自回归循环以 C++ 手工实现，仅涉及矩阵乘法、softmax 与按索引取向量等基本运算。

## 两处主要工作量

**其一是音频前端，属全新实现。** mel 特征采用标准的 WhisperFeatureExtractor 配置（128 个 mel 通道、n_fft 400、hop 160、16kHz）。该模块成熟且有现成参考实现，C++ 侧通过 reflect padding、汉宁窗、STFT、mel 滤波与对数归一化即可逐值复现。前端的卷积部分为三层 stride 为 2 的 Conv2d，每层将时间与频率维度各减半；实现上将 mel 按 100 帧分块后逐块下采样，再投影至 896 维。该部分经 pnnx 转换至 ncnn 较为顺利，Conv2d 算子均受支持，未出现 TTS 中与 mask 相关的问题。

**其二是编码器的注意力机制，存在一处极易误判的细节。** 编码器代码中包含一套 `cu_seqlens` 逻辑，将序列切分为块，表面上呈现"块内双向、块间隔离"的分块注意力形态。初始实现即按此理解编写，结果第 0 层输出即无法对齐。由于每一层均预先保存了 golden 张量，通过逐层比对迅速定位到注意力作用范围有误。进一步审阅代码后确认：该 `cu_seqlens` 仅服务于 FlashAttention-2 的 varlen 路径；而为便于数值对齐，本实现运行的是 eager 注意力，此时 attention_mask 为 None，实际执行的是**全局双向注意力**——所有位置相互可见。修正为全局注意力后，第 0 层 maxdiff 随即降至 1e-6，整个编码器降至 3e-7。

> 经验依然是：不应轻信代码的表面结构，而应以逐层 golden 张量比对定位问题——误差出现在哪一层，一目了然。

## Thinker 与 Talker 高度同构

至 Thinker 阶段方才充分体现本次工作的省力之处。其配置与 TTS 中的 Talker 几乎完全一致：同为 28 层、hidden 1024、GQA 16/8、head_dim 128、QK-norm、SwiGLU、RMSNorm、RoPE base 1e6。位置编码亦相同——标称为交错排布 `[24,20,20]` 的 mRoPE，但 ASR 场景下文本位置在三个维度上取值相同，交错最终退化为普通 RoPE，恰为 TTS 项目中已推导并验证过的情形。

因此 Thinker 的 C++ 实现基本沿用 Talker 的注意力代码，仅需修改三处：词表更换为文本词表（151936）；`tie_word_embeddings` 使输出头复用词嵌入，即以 hidden 与词嵌入做点积后取 argmax；以及将音频 embedding 注入 prompt。注入的实现较为直接：提示词中含 193 个 `<|audio_token|>` 占位符，将其嵌入替换为编码器输出的 193 行音频 embedding 即可——编码器输出维度恰为 1024，与 LLM 的 hidden 维度一致，无需额外投影。

自编码器输出起，Thinker 以 greedy 解码生成的 52 个 token 一次性全部对齐，且恰在句末生成 eos。

## 验证方法与端到端联调

验证方法与 TTS 项目一致：从 PyTorch 侧导出每一级的中间张量作为标准答案，C++ 侧每实现一级即比对一级。mel、前端、编码器、注入、生成、解码逐环对齐之后，将各模块串联为单一可执行文件，自行从头构造完整的 211 token 对话 prompt（系统头、193 个音频占位符、`assistant` 标记与语言提示），运行完整链路。最终输出的文字与 `qwen-asr` 逐字一致。

## 将前端真正运行于 ncnn

仅完成 conv 到 ncnn 的转换并不足够，还须使可执行文件实际调用 ncnn。实现上将三层 Conv2d 连同 conv_out 一并经 pnnx 导出为单一 ncnn 模型（设置 `fp16=False`——此为 TTS 项目的重要教训：默认以 fp16 存储会使每层累积约千分之一的误差，经深层网络放大后不可接受），C++ 侧通过 `ncnn::Net` 逐块执行前端，其余部分保持 C++ 实现。替换为 ncnn 前端后，端到端输出仍逐字一致。至此，推理链路中 ncnn 已实际参与计算，而非仅停留在"模型可转换"的层面。

## 当前局限

- 目前仅将计算量最大的卷积前端接入 ncnn，编码器与 Thinker 的 MLP、投影层仍以 C++ 执行。此类模块均为纯前馈结构，转换方式与前端相同，尚待逐一接入。
- 运行验证仅覆盖 Linux（x86_64）；Windows 平台通过 GitHub Actions CI 保证可编译。
- 目前打通的是英文链路；切换语言需修改 prompt 中的语言提示 token。
- Thinker 已启用 KV-cache（每层缓存 K/V，自回归复杂度由 O(T²) 降至 O(T)），但短句场景下的主要开销在于输出头对约 15 万词的词表做 argmax，该部分尚未优化。

## 经验总结

多数经验与 TTS 项目相同，可直接复用：ncnn 不建模 batch 维，故前馈算子进 ncnn、注意力以 C++ 实现；pnnx 导出必须设置 `fp16=False`；唯有 greedy 解码方可与 PyTorch 逐字对齐；数值不一致时应首先以逐层 golden 张量定位。ASR 特有的一条经验是：不应被 `cu_seqlens` 等"分块"表象误导而实现块内注意力，应首先确认实际运行的注意力后端为何——eager 模式下其行为可能即为全局注意力。

而最划算的一条经验在于：先完成一个同源模型（TTS）的移植，则第二个模型（ASR）中的大部分结构均已是熟悉的内容。

---

*致谢：模型来自 [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)；推理框架为 [Tencent/ncnn](https://github.com/Tencent/ncnn) 与 pnnx；C++ 侧 LLM 解码参考 [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm)；姊妹项目 [Qwen3-TTS-ncnn](https://github.com/caixuehe/Qwen3-TTS-ncnn)。*