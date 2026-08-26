可以。我会把这门课正式定义成：

# Rust Neural Network Inference Runtime From Scratch

目标不是“用 Rust 调一个 ML 框架”，而是从：

```text
Vec<f32>
   ↓
Tensor
   ↓
Operator
   ↓
MLP
   ↓
Tiny Transformer
   ↓
Inference Runtime
   ↓
CNN / RNN
```

一路自己搭起来。

而且主线**不写 backward、不写 autograd、不写 optimizer**。Python/PyTorch/NumPy 只允许作为“标准答案生成器”，不能进入 Rust engine。

最终你应该得到一个类似“小型 ggml / ONNX Runtime”的东西，而不仅仅是一个 GPT demo。ggml 本身就是低层 tensor library，并把 computation graph、跨硬件 backend、量化和运行期内存管理作为重要组成部分，所以很适合作为后半程的“答案本”，但前期不要照着它抄。([GitHub][1])

---

# 一、课程最终目标

做完以后，希望你的项目大概长这样：

```text
rust-infer/
│
├── src/
│   ├── tensor/
│   │   ├── storage.rs
│   │   ├── shape.rs
│   │   ├── layout.rs
│   │   └── tensor.rs
│   │
│   ├── ops/
│   │   ├── elementwise.rs
│   │   ├── reduce.rs
│   │   ├── matmul.rs
│   │   ├── softmax.rs
│   │   ├── norm.rs
│   │   └── conv.rs
│   │
│   ├── nn/
│   │   ├── linear.rs
│   │   ├── embedding.rs
│   │   ├── layernorm.rs
│   │   └── activations.rs
│   │
│   ├── models/
│   │   ├── mlp.rs
│   │   ├── tiny_gpt.rs
│   │   ├── cnn.rs
│   │   └── rnn.rs
│   │
│   ├── runtime/
│   │   ├── graph.rs
│   │   ├── planner.rs
│   │   ├── executor.rs
│   │   ├── arena.rs
│   │   └── session.rs
│   │
│   ├── backend/
│   │   ├── cpu.rs
│   │   └── kernels/
│   │
│   └── io/
│       ├── model.rs
│       └── tokenizer.rs
│
├── tests/
│   └── reference/
│
├── benches/
│
└── tools/
    └── generate_reference.py
```

课程主线可以分成：

| 阶段 | 最终产物              | 主要学习的问题                             |
| -- | ----------------- | ----------------------------------- |
| 0  | 开发/测试框架           | Rust memory model、reference testing |
| 1  | Tensor Engine     | Tensor 到底是什么                        |
| 2  | MLP               | 怎么把 operator 组成 model               |
| 3  | Tiny Transformer  | 复杂 tensor layout + attention        |
| 4  | Inference Runtime | graph、memory、kernel、KV cache、性能     |
| 5  | CNN / RNN         | 验证这个框架是不是真的通用                       |

我建议整个课程**先 CPU-only + f32**。

CUDA、f16、bf16、int8、int4 全部放在主线完成之后。

---

# Stage 0：准备工作

## Chapter 0.1 Rust 中的内存

你需要真正理解：

```text
Vec<T>
&T
&[T]
&mut [T]
Box<T>
ownership
borrow
lifetime
```

尤其是：

```rust
Vec<f32>
```

和：

```rust
&[f32]
```

之间的区别。

Rust 官方文档对 slice 的定义非常适合这里：slice 是对连续元素序列的引用，本身不拥有底层数据。这个概念以后几乎会直接变成 Tensor View 的思想。([Rust 文档][2])

### 必须阅读

Rust Book：

[Rust Book — The Slice Type](https://doc.rust-lang.org/book/ch04-03-slices.html?utm_source=chatgpt.com)

Rust `Vec<T>`：

[Rust std::vec::Vec documentation](https://doc.rust-lang.org/std/vec/struct.Vec.html?utm_source=chatgpt.com)

### 进阶

如果想了解 `Vec<T>` 底下其实是什么：

[Rustonomicon — Implementing Vec](https://doc.rust-lang.org/nomicon/vec/vec.html?utm_source=chatgpt.com)

但这个阶段**不要自己实现 Vec**。

---

# Stage 1：Tensor

这是整门课最重要的阶段之一。

不要急着 Transformer。

这一阶段的目标是最后能写：

```rust
let a = Tensor::from_vec(
    vec![1., 2., 3., 4., 5., 6.],
    &[2, 3]
);

let b = a.transpose(0, 1);

println!("{:?}", b.shape());
// [3, 2]
```

而且你知道：

> `transpose()` 到底有没有复制内存。

---

## Chapter 1.1 Storage + Shape

第一版：

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

先不要 stride。

实现：

```rust
Tensor::zeros()
Tensor::ones()
Tensor::from_vec()
Tensor::numel()
Tensor::shape()
Tensor::ndim()
Tensor::get()
Tensor::set()
```

比如：

```text
shape = [2, 3, 4]
```

你必须自己实现：

$$
(i,j,k)\rightarrow offset
$$

对于 contiguous row-major：

$$
offset=i(3\times4)+j(4)+k
$$

### 作业

实现任意 N 维 indexing：

```rust
tensor.get(&[1, 2, 3])
```

---

# Chapter 1.2 Stride

然后修改：

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}
```

这时候必须理解：

```text
shape   = [2, 3, 4]

stride  = [12, 4, 1]
```

于是：

$$
offset
=
i_0s_0+i_1s_1+\cdots+i_ns_n
$$

NumPy 官方对 stride 的定义正是“沿某个 dimension 前进一个元素时，需要跨过多少字节”，并给出了 offset 的计算方式。([NumPy][3])

### 必读

[NumPy ndarray.strides](https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html?utm_source=chatgpt.com)

---

# Chapter 1.3 View / Transpose / Reshape

这章会第一次让你觉得：

> “哦，Tensor 原来不是个二维数组。”

例如：

```text
原 Tensor

shape   = [2,3]
stride  = [3,1]

transpose

shape   = [3,2]
stride  = [1,3]
```

**data 一点都没动。**

于是你会真正理解：

```text
Tensor
=
Storage
+
Shape
+
Stride
+
Offset
```

更合理的设计甚至逐渐变成：

```rust
struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}
```

你需要实现：

```rust
reshape()
transpose()
slice()
contiguous()
is_contiguous()
```

PyTorch 的 Tensor View 文档非常值得反复看。它明确区分了 view 和实际数据复制，并指出 `transpose()` 经常产生 non-contiguous tensor。([PyTorch Docs][4])

### 必读

[PyTorch — Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html?utm_source=chatgpt.com)

---

# Chapter 1.4 Elementwise Operator

实现：

```text
add
sub
mul
div

exp
sqrt
tanh

relu
gelu
silu
```

第一版要求：

```rust
add(&a, &b)
```

要求 shape 完全一致。

之后再做 broadcasting。

---

# Chapter 1.5 Broadcasting

实现：

```text
[2,3,4]
+
[      4]

→ [2,3,4]
```

然后：

```text
[8,32,128,64]
+
[         64]
```

这以后对 bias、attention、norm 都非常重要。

你必须能自己解释：

```text
broadcast 到底有没有复制数据？
```

第一版可以真的 loop。

后面再考虑 stride=0 一类的 view representation。

---

# Chapter 1.6 Reduction

实现：

```rust
sum()
mean()
max()
```

支持：

```rust
x.sum(axis)
```

例如：

```text
[2,3,4]

sum(dim=2)

→ [2,3]
```

Softmax、LayerNorm、RMSNorm 都依赖 reduction。

---

# Chapter 1.7 MatMul

第一版只支持：

```text
[M,K] × [K,N]
```

最笨版本：

```rust
for i in 0..m {
    for j in 0..n {
        let mut sum = 0.0;

        for k in 0..K {
            sum += a[i * K + k] * b[k * N + j];
        }

        c[i * N + j] = sum;
    }
}
```

然后把 loop 顺序改成：

```text
ijk
ikj
jik
```

测试性能区别。

**现在不要 SIMD。**

现在的目标是：

> 正确。

---

## Stage 1 验收

至少满足：

```text
✓ N-D Tensor
✓ indexing
✓ shape
✓ stride
✓ slice
✓ reshape
✓ transpose
✓ contiguous
✓ broadcasting
✓ reduction
✓ matmul
```

然后写：

```text
Rust output
    ↓ compare
NumPy output
```

随机生成几百组 shape 测。

只要误差：

```text
abs(a - b) < 1e-5
```

基本就可以继续。

---

# Stage 2：MLP

这一步非常短，但非常重要。

它第一次把你的：

```text
Tensor Library
```

变成：

```text
Neural Network Engine
```

---

# Chapter 2.1 Linear

数学：

$$
Y=XW+b
$$

实现：

```rust
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}
```

然后：

```rust
impl Linear {
    fn forward(&self, x: &Tensor) -> Tensor;
}
```

底层只能调用：

```text
matmul
add
```

---

# Chapter 2.2 Activation

实现：

```text
ReLU
GELU
SiLU
Tanh
Sigmoid
```

于是：

```rust
Linear
 ↓
GELU
 ↓
Linear
```

就是一个 MLP。

---

# Chapter 2.3 Model abstraction

第一版不要立刻设计：

```rust
trait Module
```

甚至可以直接：

```rust
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(x);
        let x = gelu(&x);
        self.fc2.forward(&x)
    }
}
```

为什么？

因为现在你还不知道：

> “Module 应该抽象成什么。”

等 Transformer 写完再抽象，设计通常会好得多。

---

# Chapter 2.4 Reference Model

这时候可以用 PyTorch：

```python
model = nn.Sequential(
    nn.Linear(...),
    nn.GELU(),
    nn.Linear(...)
)
```

随机初始化。

把：

```text
weights
input
expected output
```

导出来。

Rust 读取后：

```text
Rust MLP
vs
PyTorch MLP
```

要求逐元素对齐。

注意：

> PyTorch 在这里是测试 oracle，不是你的实现依赖。

---

## Stage 2 验收

你应该能够做到：

```text
input.bin
weights.bin
    ↓
Rust MLP
    ↓
output

≈

PyTorch output
```

到这里才开始 Transformer。

---

# Stage 3：Tiny Transformer

这里我建议分两次。

## Transformer V1

**GPT-2 风格**

```text
Embedding
+
Position Embedding
       ↓
LayerNorm
       ↓
Multi Head Attention
       ↓
Residual
       ↓
LayerNorm
       ↓
MLP + GELU
       ↓
Residual
```

因为简单。

## Transformer V2

再升级到 Llama 风格：

```text
RoPE
RMSNorm
SwiGLU
```

这样学习曲线舒服很多。

---

# Chapter 3.1 Transformer 数学

先读原论文，但不用每节都啃。

核心只看：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

以及：

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Transformer 原论文的核心贡献就是用 attention 构建架构，而不依赖 recurrence/convolution。([arXiv][5])

### 必读

[Attention Is All You Need](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com)

但是比论文更适合实际写代码的是：

[Karpathy — build-nanogpt](https://github.com/karpathy/build-nanogpt?utm_source=chatgpt.com)

Karpathy 的代码直接展示了：

```text
[B,T,C]

→ QKV
→ [B,T,H,D]
→ transpose
→ [B,H,T,D]
```

以及 attention 后怎么重新组装。([GitHub][6])

---

# Chapter 3.2 Embedding

Embedding 本质上就是：

```text
weight[vocab_size][hidden_size]
```

token：

```text
token_id = 37
```

就是取：

```text
weight[37]
```

所以实现：

```rust
embedding(weight, token_ids)
```

本质上是：

> gather。

---

# Chapter 3.3 Positional Embedding

V1 直接实现 GPT 风格：

```text
token_embedding
+
position_embedding
```

例如：

```text
token 0 → position 0
token 1 → position 1
...
```

不要现在上 RoPE。

GPT-2 是非常合适的 decoder-only reference architecture。([OpenAI CDN][7])

---

# Chapter 3.4 LayerNorm

实现：

$$
\mu=\frac1N\sum_i x_i
$$

$$
\sigma^2=\frac1N\sum_i(x_i-\mu)^2
$$

$$
y_i=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
\gamma_i+\beta_i
$$

这里会第一次真正检验你的：

```text
reduce
broadcast
elementwise
```

有没有设计好。

---

# Chapter 3.5 Softmax

一定做 stable softmax：

$$
softmax(x_i)=
\frac{e^{x_i-m}}
{\sum_j e^{x_j-m}}
$$

其中：

$$
m=\max_jx_j
$$

否则大数容易 overflow。

---

# Chapter 3.6 Single-Head Attention

先只支持：

```text
B = 1
H = 1
```

不要一上来 multi-head。

实现：

```text
X
 ↓
Q K V
 ↓
QK^T
 ↓
scale
 ↓
causal mask
 ↓
softmax
 ↓
× V
```

并且每一步都 dump tensor。

例如：

```text
q.bin
k.bin
v.bin
score.bin
prob.bin
output.bin
```

和 PyTorch 一项项比较。

---

# Chapter 3.7 Multi-Head Attention

这时候第一次强迫你处理：

```text
[B,T,C]

reshape

[B,T,H,D]

transpose

[B,H,T,D]
```

这里是 Tensor Stage 的真正期末考试。

如果你的 stride/view API 设计不好，这里会开始爆炸。

需要增加：

```text
batched matmul
```

支持：

```text
[B,H,T,D]
×
[B,H,D,T]

→

[B,H,T,T]
```

---

# Chapter 3.8 Transformer Block

拼起来：

```text
x
 │
 ├── LayerNorm
 │      ↓
 │   Attention
 │      ↓
 └──── Add
        │
        ├── LayerNorm
        │       ↓
        │      MLP
        │       ↓
        └───── Add
```

然后：

```rust
struct TransformerBlock {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    mlp: Mlp,
}
```

---

# Chapter 3.9 Tiny GPT

例如：

```text
vocab = 256
hidden = 128
heads = 4
layers = 2
context = 128
```

参数量很小。

Rust 不训练。

用 Python：

```text
TinyStories / Shakespeare
       ↓
train TinyGPT
       ↓
export weights
       ↓
Rust inference
```

TinyStories 特别适合这种实验，因为论文表明极小模型、甚至只有一个 Transformer block 的模型也可以在这个简单数据分布上产生有意义输出。([arXiv][8])

[TinyStories paper](https://arxiv.org/abs/2305.07759?utm_source=chatgpt.com)

---

# Chapter 3.10 Generation

实现：

```text
prompt
 ↓
tokenize
 ↓
forward
 ↓
last logits
 ↓
temperature
 ↓
softmax
 ↓
sample
 ↓
next token
 ↓
repeat
```

实现：

```text
greedy
temperature
top-k
```

top-p 可以晚一点。

Tokenizer 第一版直接：

```text
char-level
```

或者：

```text
byte-level
```

**不要现在自己实现 BPE。**

那是另一条支线。

---

# Chapter 3.11 Llama Block

Tiny GPT 完成后再加入：

```text
LayerNorm → RMSNorm
PositionEmbedding → RoPE
GELU MLP → SwiGLU
```

这时候去读：

RoPE：([arXiv][9])
[RoFormer / RoPE paper](https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com)

RMSNorm：([arXiv][10])
[RMSNorm paper](https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com)

SwiGLU：([arXiv][11])
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202?utm_source=chatgpt.com)

最后推荐一个非常重要的“考试”：

> 用 Rust 重写 `llama2.c` 的 inference。

`llama2.c` 就是一个非常小、教育导向的 Llama inference reference，项目明确强调 minimal C implementation，因此非常适合在你已经理解 Transformer 后拿来对答案。([GitHub][12])

[Karpathy — llama2.c](https://github.com/karpathy/llama2.c?utm_source=chatgpt.com)

---

# Stage 3 验收标准

这是非常重要的一关。

要求：

```text
同一组权重
同一组 token

PyTorch
   ↓
logits

≈

Rust
   ↓
logits
```

不是：

> “生成出来看起来挺像英语。”

而是：

$$
|logit_{rust}-logit_{torch}| < \epsilon
$$

如果这里做不到，不允许进入 Runtime。

---

# Stage 4：Inference Runtime

到这里课程性质会突然改变。

之前你研究的是：

> Neural Network

从现在开始研究的是：

> Systems。

整个项目真正有意思的部分其实从这里开始。

---

# Chapter 4.1 Benchmark Infrastructure

第一件事不是优化。

而是：

> 测量。

记录：

```text
MatMul GFLOP/s
Attention latency
prefill latency
decode latency
tokens/s
peak memory
allocation count
```

Rust 可以使用 Criterion 做微基准，它专门用于统计式 microbenchmark 和性能回归检测。([Docs.rs][13])

[Criterion.rs documentation](https://docs.rs/criterion/latest/criterion/?utm_source=chatgpt.com)

---

# Chapter 4.2 Eager → Graph

目前代码大概是：

```rust
let q = matmul(x, wq);
let k = matmul(x, wk);
let v = matmul(x, wv);
```

现在第一次引入：

```rust
enum Op {
    MatMul,
    Add,
    Gelu,
    Softmax,
    ...
}
```

以及：

```rust
struct Node {
    op: Op,
    inputs: Vec<TensorId>,
    output: TensorId,
}
```

最后：

```rust
struct Graph {
    nodes: Vec<Node>,
}
```

于是：

```text
Tensor
 ↓
Operator
 ↓
Graph
 ↓
Executor
```

---

# Chapter 4.3 Executor

第一版：

```rust
for node in graph.nodes {
    execute(node);
}
```

实现：

```rust
struct Executor;
```

目标是把：

```text
模型定义
```

和：

```text
模型执行
```

拆开。

---

# Chapter 4.4 Tensor Lifetime

假设：

```text
A ──┐
    MatMul → C → GELU → D → ...
B ──┘
```

C 用完以后：

> 它那块内存还需要存在吗？

这就引出：

```text
tensor lifetime
```

你开始记录：

```text
first_use
last_use
size
```

这就是 Memory Planner 的基础。

---

# Chapter 4.5 Arena Allocator

最笨版本，每个 op：

```rust
Vec::with_capacity(...)
```

然后释放。

这会导致很多 allocation。

改成：

```text
             Arena
┌────────────────────────────┐
│ A │       B │ C │   D     │
└────────────────────────────┘
```

Runtime 启动前：

```text
Graph
 ↓
Tensor lifetime analysis
 ↓
Memory plan
 ↓
Allocate one arena
```

执行时：

```text
0 runtime malloc
```

这其实是成熟 inference runtime 很重要的思想。TFLite 对 intermediate tensors 做 lifetime 分析和 buffer reuse，CPU 路径可以预先规划一块大型 memory arena 再按 offset 使用。([TensorFlow 博客][14])

[TensorFlow Lite — Runtime Memory Optimization](https://blog.tensorflow.org/2020/10/optimizing-tensorflow-lite-runtime.html?utm_source=chatgpt.com)

Glow 也是非常值得看的系统论文，它把 graph lowering、static memory allocation、copy elimination 都放进编译流程。([arXiv][15])

[Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/abs/1805.00907?utm_source=chatgpt.com)

---

# Chapter 4.6 Backend / Kernel

现在把：

```rust
matmul()
```

拆成：

```text
Operator semantics
      ↓
Kernel implementation
```

例如：

```rust
trait Backend {
    fn matmul(...);
    fn add(...);
    fn softmax(...);
}
```

第一版：

```text
CpuBackend
```

以后：

```text
CpuBackend
CudaBackend
MetalBackend
```

ONNX Runtime 的 Execution Provider 思想很值得参考：runtime 可以把不同节点交给不同硬件/provider 执行。([ONNX Runtime][16])

[ONNX Runtime — Execution Providers](https://onnxruntime.ai/docs/execution-providers/?utm_source=chatgpt.com)

---

# Chapter 4.7 Graph Optimization

开始做：

```text
constant folding
dead node removal
operator fusion
```

比如：

```text
MatMul
 ↓
Add
 ↓
GELU
```

未来可以变成：

```text
FusedLinearGelu
```

ONNX Runtime 官方把 graph optimizations 分为 basic、extended、layout 等层级，并包含 constant folding、节点消除、fusion 等典型优化。([ONNX Runtime][17])

[ONNX Runtime — Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html?utm_source=chatgpt.com)

但课程里**只做两三个简单 pass**。

不要自己搞完整 compiler。

---

# Chapter 4.8 KV Cache

终于进入真正的 LLM inference。

没有 KV Cache：

第 \(t\) 步：

```text
token 0 ... token t
       ↓
重新算全部 K,V
```

有 KV cache：

```text
K0
K1
K2
...
Kt
```

过去的：

```text
K/V
```

全部保存。

每个 decode step 只计算新 token。

于是把 inference 分成：

```text
prefill
   ↓
KV Cache
   ↓
decode
   ↓
decode
   ↓
decode
```

并设计：

```rust
struct KvCache {
    keys: ...,
    values: ...,
    len: usize,
}
```

这时候你的 runtime 第一次拥有：

> persistent state。

---

# Chapter 4.9 Prefill vs Decode

你需要认识一个非常重要的区别。

Prefill：

```text
[B,T,D]
```

往往是：

> 大矩阵计算。

Decode：

```text
[B,1,D]
```

往往变成：

> 小矩阵 + KV memory access。

于是：

```text
同一个 Transformer
```

实际上有两套非常不同的性能特征。

这对以后理解 LLM serving 至关重要。

---

# Chapter 4.10 MatMul Optimization

现在终于回来优化：

```text
naive MatMul
```

顺序：

```text
ijk
 ↓
ikj
 ↓
tiling
 ↓
cache blocking
 ↓
SIMD
 ↓
multithreading
```

Rust 的 architecture-specific SIMD intrinsics 在 `std::arch` / `core::arch` 下提供，但它们是硬件相关的，不是 portable abstraction。([Rust 文档][18])

[Rust std::arch documentation](https://doc.rust-lang.org/nightly/std/arch/index.html?utm_source=chatgpt.com)

这里第一次认真研究：

```text
cache line
L1
L2
L3
memory bandwidth
SIMD width
```

---

# Chapter 4.11 Model File

最开始你自己的格式可以极其简单：

```text
magic
version
tensor count

tensor name
shape
dtype
data offset

...

tensor data
```

然后再读 Safetensors。

Safetensors 本身就是面向 tensor storage 的简单安全格式，并强调 fast/zero-copy loading。([Hugging Face][19])

[Hugging Face — Safetensors Format](https://huggingface.co/docs/safetensors/index?utm_source=chatgpt.com)

再之后才看：

GGUF。

GGUF 特别适合 inference，设计目标包括单文件部署、扩展 metadata、快速 loading，以及 mmap 兼容。([GitHub][20])

[GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md?utm_source=chatgpt.com)

---

# Chapter 4.12 mmap

把：

```text
read()
 ↓
copy
 ↓
Vec
```

改成：

```text
mmap(model.gguf)
```

理解：

```text
virtual memory
page
page fault
file-backed memory
```

这已经进入标准 systems territory 了。

---

# Chapter 4.13 Quantization

主线完成后再做：

```text
f32
 ↓
f16 / bf16
 ↓
int8
 ↓
int4
```

第一版实现最简单：

$$
q=round(x/s)
$$

$$
x\approx q\cdot s
$$

然后研究：

```text
per-tensor
per-channel
group-wise
```

不要一开始就抄 GGML 的各种：

```text
Q4_K
Q5_K
IQ*
```

---

# Chapter 4.14 FlashAttention —— 只读论文

到这里再读 FlashAttention 会完全不一样。

它的关键不是：

> 一个新的 Attention 数学公式。

而是：

> **通过 tiling 降低 HBM 与片上 SRAM 之间的数据搬运。**

论文明确从 IO-aware 的角度重新组织 exact attention。([arXiv][21])

[FlashAttention paper](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

第一遍：

**不实现。**

把算法画出来就行。

---

# Chapter 4.15 LLM Serving —— 只做扩展

等单 request inference 完整跑通，才学习：

```text
batching
continuous batching
request scheduling
KV cache allocation
```

这时读 vLLM / PagedAttention。

PagedAttention 的核心问题就是 KV cache 动态增长会产生 fragmentation 和 wasted memory，因此借鉴虚拟内存 paging 思想管理 KV blocks。([arXiv][22])

[PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com)

这已经属于：

> **Inference Serving System**

而不是基础 runtime。

---

# Runtime 阶段最重要的参考项目

我建议阅读顺序是：

```text
llama2.c
   ↓
ggml
   ↓
llama.cpp
   ↓
ONNX Runtime
   ↓
TVM
```

不要反过来。

因为如果直接从 TVM 开始，你看到的很可能是：

```text
IR
lowering
schedule
codegen
target
pass
```

然后完全不知道这些 abstraction 为什么存在。

TVM 论文强调 graph-level/operator-level optimization、operator fusion 和跨硬件 backend，这非常适合在你已经写过自己的 executor 后再看。([arXiv][23])

[TVM paper](https://arxiv.org/abs/1802.04799?utm_source=chatgpt.com)

[TVM Architecture Documentation](https://tvm.apache.org/docs/arch/index.html?utm_source=chatgpt.com)

---

# Stage 4 最终验收

我会要求你的 Runtime 达到：

```text
Model File
    ↓
Loader
    ↓
Tensor / Weight
    ↓
Graph
    ↓
Memory Planner
    ↓
Execution Plan
    ↓
CPU Backend
    ↓
Kernel
```

而 LLM：

```text
Prompt
   ↓
Prefill
   ↓
KV Cache
   ↓
Decode Loop
   ↓
Sampling
```

并且至少输出：

```text
model load time
prefill latency
TTFT
decode ms/token
tokens/s
peak memory
arena size
```

做到这里，项目已经非常像一个真正的 inference engine 了。

---

# Stage 5：CNN / RNN 通用性测试

这里不是为了让你学计算机视觉或者序列模型。

目的是：

> **验证你的 Runtime 是不是偷偷写成了 GPT Runtime。**

---

# Chapter 5.1 RNN

先 RNN，因为简单。

实现 vanilla RNN：

$$
h_t=
\tanh(
x_tW_x+h_{t-1}W_h+b
)
$$

需要的 op 几乎已经都有：

```text
MatMul
Add
Tanh
```

唯一的新东西是：

```text
state
```

也就是：

```rust
for t in 0..sequence {
    h = cell.forward(x[t], h);
}
```

这会测试：

```text
Tensor slicing
stateful execution
loop
buffer reuse
```

### 推荐教材

D2L 的 RNN 章节有从序列建模一直到从零实现 RNN 的完整路线。([深度学习入门][24])

[Dive into Deep Learning — Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html?utm_source=chatgpt.com)

之后可以选做：

```text
LSTM
GRU
```

但不是必须。

---

# Chapter 5.2 CNN

实现：

```text
Conv2D
ReLU
MaxPool
Flatten
Linear
```

然后：

```text
LeNet
```

第一版固定：

```text
NCHW
```

输入：

$$
[N,C_{in},H,W]
$$

kernel：

$$
[C_{out},C_{in},K_h,K_w]
$$

写最暴力版：

```text
N
 Cout
  Hout
   Wout
    Cin
     Kh
      Kw
```

七层 loop。

不要怕丑。

这是学习：

```text
layout
stride
padding
channel
kernel
```

最好的实验之一。

### 推荐资料

Stanford CS231n 的 ConvNet notes 对 convolution、pooling、layer sizing 和经典结构的讲解非常清晰。([CS231n][25])

[Stanford CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/?utm_source=chatgpt.com)

D2L：

[Dive into Deep Learning — CNN](https://d2l.ai/chapter_convolutional-neural-networks/index.html?utm_source=chatgpt.com)

它从 padding、stride、多 channel 一直到 LeNet 都有覆盖。([深度学习入门][26])

---

# 这一阶段真正的验收标准

最关键的一条不是：

> CNN accuracy 多高。

而是：

**加入 CNN 时你有没有修改 Tensor Core？**

理想情况：

```text
Tensor        不改
Executor      基本不改
Arena         不改
Backend       加 Conv kernel
Ops           加 Conv2D / Pool
Models        加 LeNet
```

RNN 同样：

```text
Tensor        不改
Backend       基本不改
Ops           几乎不加
Models        加 RNN
Runtime       支持 persistent state
```

如果做到了：

> 说明你的 abstraction 基本成功了。

如果加入 CNN 以后发现：

```text
Tensor {
    ...
    transformer_special_xxx
}
```

那就是架构开始歪了。

---

# 整门课我会这样安排

如果按章节而不是日历，我建议：

| Part | 内容                         |   重要性 |
| ---- | -------------------------- | ----: |
| 0    | Rust memory / testing      |   ★★★ |
| 1    | Storage / Shape            | ★★★★★ |
| 2    | Stride / View              | ★★★★★ |
| 3    | Elementwise / Broadcast    |  ★★★★ |
| 4    | Reduction / MatMul         | ★★★★★ |
| 5    | MLP                        |   ★★★ |
| 6    | Embedding / Norm / Softmax |  ★★★★ |
| 7    | Single-head Attention      | ★★★★★ |
| 8    | Multi-head Attention       | ★★★★★ |
| 9    | Tiny GPT                   | ★★★★★ |
| 10   | Generation                 |  ★★★★ |
| 11   | Llama block                |  ★★★★ |
| 12   | Benchmark                  | ★★★★★ |
| 13   | Graph / Executor           | ★★★★★ |
| 14   | Lifetime / Arena           | ★★★★★ |
| 15   | Backend / Kernel           | ★★★★★ |
| 16   | KV Cache                   | ★★★★★ |
| 17   | MatMul optimization        | ★★★★★ |
| 18   | File format / mmap         |  ★★★★ |
| 19   | Quantization               |  ★★★★ |
| 20   | CNN / RNN                  |   ★★★ |

---

# 几条“课程纪律”

我会比较强硬地规定这几条：

**第一，engine 主线禁止：**

```text
Candle
Burn
tch
ndarray
nalgebra
BLAS
MKL
```

直到你完成 naive MatMul。

**第二，允许测试使用：**

```text
Python
NumPy
PyTorch
```

因为：

```text
自己实现 ≠ 自己猜答案
```

你必须有 reference。

**第三，不写 backward。**

主线只有：

```text
forward inference
```

Autograd 独立作为可选支线。

**第四，先正确，后性能。**

严格按照：

```text
correct
  ↓
profile
  ↓
understand
  ↓
optimize
```

禁止：

```text
代码还没对
 ↓
开始 SIMD
 ↓
开始 unsafe
 ↓
开始多线程
```

**第五，任何优化必须 benchmark。**

不要：

> “这个 loop 看起来 cache friendly，所以肯定快。”

而是：

```text
before: 3.7 ms
after:  2.2 ms
speedup: 1.68×
```

**第六，前半段不要看 ggml 源码答案。**

到 Runtime 阶段再打开：

[ggml repository](https://github.com/ggml-org/ggml?utm_source=chatgpt.com)

那时候你会产生很多非常有价值的瞬间：

> “哦，原来我遇到的这个问题，他们是这么解决的。”

而不是一开始：

> “这个 struct 为什么长这样，我照着抄好了。”

---

## 最终你真正学习到的不是 Transformer

我认为这条课程最漂亮的地方其实是它最后会形成三条知识线：

```text
        ML
        │
MLP → Attention → Transformer
        │
        │
        ▼
      Tensor
        │
        │
        ▼
      Systems
        │
Memory Layout
Cache
SIMD
Threads
Allocator
mmap
        │
        ▼
      Runtime
        │
Graph
Planner
Executor
Backend
Kernel
Scheduling
```

到了后半段，你再去看：

```text
PyTorch
Candle
ggml
llama.cpp
ONNX Runtime
TVM
TensorRT
vLLM
```

关注点就不会再是：

> “这个框架 API 怎么用？”

而会变成：

> “它的 Tensor representation 是什么？”
>
> “view 如何表示？”
>
> “graph 什么时候构建？”
>
> “memory planner 怎么算 lifetime？”
>
> “kernel dispatch 在哪里？”
>
> “prefill/decode 为什么走不同路径？”
>
> “KV cache 谁管理？”
>
> “为什么这里需要 fusion？”
>
> “为什么这段代码 memory-bound？”

这才是我认为这门课真正应该训练出来的能力。

**如果正式开始，我建议下一步就从 Stage 1 的 Chapter 1.1 开始，不再继续看 Transformer 资料。第一个小目标就是设计 `Tensor { Vec<f32>, shape }`，实现 N 维 indexing，然后我们再一步一步把 stride 引进来。**

[1]: https://github.com/ggml-org/ggml?utm_source=chatgpt.com "GitHub - ggml-org/ggml: Tensor library for machine learning · GitHub"
[2]: https://doc.rust-lang.org/book/ch04-03-slices.html "The Slice Type - The Rust Programming Language"
[3]: https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html?utm_source=chatgpt.com "numpy.ndarray.strides — NumPy v2.6.dev0 Manual"
[4]: https://docs.pytorch.org/docs/stable/tensor_view.html?utm_source=chatgpt.com "Tensor Views — PyTorch 2.13 documentation"
[5]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"
[6]: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py?utm_source=chatgpt.com "build-nanogpt/train_gpt2.py at master · karpathy/build-nanogpt · GitHub"
[7]: https://cdn.openai.com/better-language-models/language-models.pdf?utm_source=chatgpt.com "Language Models are Unsupervised Multitask Learners"
[8]: https://arxiv.org/abs/2305.07759?utm_source=chatgpt.com "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
[9]: https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com "RoFormer: Enhanced Transformer with Rotary Position Embedding"
[10]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[11]: https://arxiv.org/abs/2002.05202?utm_source=chatgpt.com "GLU Variants Improve Transformer"
[12]: https://github.com/karpathy/llama2.c?utm_source=chatgpt.com "GitHub - karpathy/llama2.c: Inference Llama 2 in one file of pure C · GitHub"
[13]: https://docs.rs/criterion/latest/criterion/?utm_source=chatgpt.com "criterion - Rust"
[14]: https://blog.tensorflow.org/2020/10/optimizing-tensorflow-lite-runtime.html?utm_source=chatgpt.com "Optimizing TensorFlow Lite Runtime Memory — The TensorFlow Blog"
[15]: https://arxiv.org/abs/1805.00907?utm_source=chatgpt.com "Glow: Graph Lowering Compiler Techniques for Neural Networks"
[16]: https://onnxruntime.ai/docs/execution-providers/ "Execution Providers | onnxruntime"
[17]: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html?utm_source=chatgpt.com "Graph optimizations | onnxruntime"
[18]: https://doc.rust-lang.org/nightly/std/arch/index.html?utm_source=chatgpt.com "std::arch - Rust"
[19]: https://huggingface.co/docs/safetensors/index?utm_source=chatgpt.com "Safetensors · Hugging Face"
[20]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md?plain=1&utm_source=chatgpt.com "ggml/docs/gguf.md at master · ggml-org/ggml · GitHub"
[21]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[22]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[23]: https://arxiv.org/abs/1802.04799?utm_source=chatgpt.com "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"
[24]: https://www.d2l.ai/?utm_source=chatgpt.com "Dive into Deep Learning — Dive into Deep Learning 1.0.3 documentation"
[25]: https://cs231n.github.io/convolutional-networks/?utm_source=chatgpt.com "CS231n Deep Learning for Computer Vision"
[26]: https://d2l.ai/chapter_convolutional-neural-networks/index.html?utm_source=chatgpt.com "7. Convolutional Neural Networks — Dive into Deep Learning 1.0.3 documentation"

