# 把 Qwen3-ASR 搬进 ncnn：一次纯 C++ 的端到端复现

做完 [Qwen3-TTS 的移植](https://github.com/caixuehe/Qwen3-TTS-ncnn) 之后，我接着把 Qwen3-ASR（0.6B）也搬到了 ncnn 上。方向正好反过来：上一个是文字进、语音出，这一个是语音进、文字出。最后同样是一个 C++ 可执行文件，喂一段音频进去，吐出转写文本，推理只依赖 ncnn，不再需要 PyTorch。

有了 TTS 的底子，这次快了很多——后面会讲，ASR 里那个自回归的语言模型几乎就是 TTS 里 Talker 的翻版，代码基本照搬。真正要从头写的只有音频前端。先把结果摆这儿：

| 环节 | 和 PyTorch 的差距 |
|---|---|
| Whisper mel 特征 | maxdiff 1.9e-5 |
| Conv2d 前端 → 编码器输入 | maxdiff 4e-6 |
| 音频编码器 → 音频 embedding | maxdiff 3e-7 |
| 音频注入 prompt → 输入 embedding | maxdiff 0（精确） |
| Thinker LLM 生成的 token | **52/52 完全一致** |
| BPE 解码 | 精确 |
| **端到端 音频 → 文字** | **与 PyTorch 逐字一致** |
| Conv 前端（pnnx→ncnn，fp16=False） | maxdiff 1.3e-5 |

代码在仓库里。样例音频用的是 `librosa.example("libri1")`（一段 LibriSpeech 英文）。

## 要移植的是什么

Qwen3-ASR 是 Qwen-Omni 那套 "Thinker" 结构的音频-LLM。拆开看是一条链：

```
音频 →[Whisper mel]→ mel →[音频编码器]→ 音频 embedding →[塞进对话 prompt 的占位符]→ 输入 embedding
     →[Thinker：28 层自回归 LLM，greedy]→ 文本 token →[byte-level BPE 解码]→ 文字
```

前半段是新东西：Whisper 风格的 mel 特征、一个卷积下采样 + 18 层的音频编码器。后半段的 Thinker 是个标准的 Qwen3 自回归解码器——它吃下音频 embedding 和一段提示词，逐个 token 地生成转写。

## 一个从 TTS 直接搬过来的判断：ncnn 没有 batch 维

这条我在 TTS 项目里就吃透了：ncnn 不建模 batch 维，而 HuggingFace 的注意力和相关算子到处依赖跨 batch 的广播和 unbind，pnnx 整块转过去，一到 ncnn 就跑不起来。所以分工照旧：卷积、MLP、投影这些前馈算子交给 pnnx 转 ncnn；注意力和自回归循环用 C++ 手写，只用到矩阵乘、softmax 和按下标取向量这几样。

## 真正花时间的两件事

**第一件是音频前端，全是新的。** mel 用的是标准 WhisperFeatureExtractor（128 个 mel、n_fft 400、hop 160、16kHz），这东西成熟、有现成参考，C++ 里 reflect padding + 汉宁窗 + STFT + mel 滤波 + 对数归一就能逐值复现。前端的卷积部分是三层 Conv2d（stride 2，每层把时间和频率各砍一半），把 mel 按 100 帧切块后逐块下采样，再投影到 896 维。这块 pnnx 转 ncnn 很顺，conv2d 全支持，没有 TTS 里那种 mask 的坑。

**第二件是编码器的注意力，一个差点看走眼的地方。** 编码器代码里有一套 `cu_seqlens`，把序列切成块，看着像是块内双向、块间不通的分块注意力。我一开始就照这个实现了，结果第 0 层输出就对不上。好在我给每一层都 dump 了 golden，一比就定位到是注意力的范围错了。翻代码才明白：那套 `cu_seqlens` 是给 FlashAttention-2 的 varlen 路径用的；而我为了能对齐、跑的是 eager 注意力，此时 attention_mask 是 None，实际是**全局双向注意力**——所有位置互相都能看到。改成全局注意力后，第 0 层 maxdiff 立刻降到 1e-6，整个编码器 3e-7。

> 教训还是那句：别信代码表面结构，拿逐层 golden 对，错在哪一层一目了然。

## Thinker 几乎是 Talker 的翻版

到了 Thinker 我才发现这次为什么这么省事。它的配置和 TTS 里的 Talker 几乎一模一样：同样 28 层、hidden 1024、GQA 16/8、head_dim 128、QK-norm、SwiGLU、RMSNorm、RoPE base 1e6。连位置编码都一样——它标称是 mRoPE 交错 `[24,20,20]`，但 ASR 的文本位置在三个维度上取值相同，交错到最后退化成普通 RoPE，正是我在 TTS 里已经推过、验过的那个情形。

所以 Thinker 的 C++ 基本是把 Talker 的注意力搬过来，改三处：词表换成文本的 151936、`tie_word_embeddings` 让输出头复用词嵌入（拿 hidden 和词嵌入做点积取 argmax）、以及把音频 embedding 注入 prompt。注入这步很直接：提示词里有 193 个 `<|audio_token|>` 占位符，把它们的嵌入换成编码器输出的 193 行音频 embedding 就行——因为编码器输出维度正好是 1024，和 LLM 的 hidden 对齐，不用再投影。

从编码器输出起，Thinker greedy 生成的 52 个 token 一次就全对，末尾正好停在 eos。

## 验证方法，和端到端

方法和 TTS 一样朴素：从 PyTorch 把每一级的中间张量 dump 出来当标准答案，C++ 实现一级、对一级。mel、前端、编码器、注入、生成、解码，逐环对齐之后，我把它们串成一个可执行文件，自己从头构造那段 211 token 的对话 prompt（系统头 + 193 个音频占位 + `assistant` + 语言提示），走完整条链。输出的文字和 `qwen-asr` 逐字一致。

## 把前端真正跑在 ncnn 上

光把 conv 转成 ncnn 还不够，得让可执行文件真的调它。我把三层 Conv2d 加上 conv_out 一起 pnnx 导出成一个 ncnn 模型（`fp16=False`——这是 TTS 的血泪教训，默认存 fp16 会让每层积累千分之一误差，深栈放大到不能用），C++ 里用 `ncnn::Net` 逐块跑前端，其余保持 C++。换上 ncnn 前端之后，端到端输出仍然逐字一致。这样这条推理链里就真有 ncnn 在干活，而不只是"转出来放着"。

## 还没做到的地方

- 目前只把最重的卷积前端接进了 ncnn，编码器和 Thinker 的 MLP / 投影还在 C++ 里跑。它们是纯前馈，转法和前端一样，只是还没逐个接回去。
- 只在 Linux（x86_64）验证过运行；Windows 走 GitHub Actions CI 保证能编译。
- 打通的是英文这条路；换语言要改 prompt 里的语言提示 token。
- Thinker 上了 KV-cache（每层缓存 K/V，自回归从 O(T²) 降到 O(T)），但短句下真正的大头是输出头对 15 万词表的 argmax，这块还没优化。

## 几条经验

大部分和 TTS 是同一套，能直接复用：ncnn 无 batch 维就前馈进 ncnn、注意力进 C++；pnnx 一定要 `fp16=False`；greedy 才谈得上和 PyTorch 逐字对齐；数值对不上先拿逐层 golden 定位。ASR 特有的一条是：别被 `cu_seqlens` 这种"分块"的表象骗了去写块内注意力，先确认实际跑的注意力后端到底是什么——eager 下它可能就是全局的。

最省事的一条其实是：先做完一个同源模型（TTS），第二个（ASR）里一大半结构都是老朋友。

---

*致谢：模型来自 [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)；推理框架是 [Tencent/ncnn](https://github.com/Tencent/ncnn) 和 pnnx；C++ 侧 LLM 解码参考 [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm)；姊妹项目 [Qwen3-TTS-ncnn](https://github.com/caixuehe/Qwen3-TTS-ncnn)。*
