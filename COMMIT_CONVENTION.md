# Commit Message 规范

本规范将提交分为两类：进入 `main` 的里程碑提交，以及功能分支上的开发过程提交。

- `main` 只保留经过 squash 的完整成果，便于按课程进度阅读历史。
- 功能分支使用粒度更小的过程提交，便于开发期间定位和撤销变更；squash 后无需保留这些提交。
- 所有 commit message 的标题都使用英文，并以大写字母开头。

本规范从合入 `main` 后的新提交开始执行，不改写既有 Git 历史。

## 1. 通用格式

```text
<Prefix>: <Subject>

[Body]

[Footer]
```

要求：

1. `Prefix` 必须采用本文定义的形式，首字母大写。
2. `Subject` 必须以大写字母开头，使用祈使语气描述本次提交完成的动作。
3. 标题建议不超过 72 个字符，末尾不加句号。
4. 一次提交只表达一个逻辑变更。
5. 需要补充背景时，在空行后的 `Body` 中说明为什么修改以及必要的设计取舍，不重复代码本身。
6. 关联 Issue 或包含破坏性变更时，在 `Footer` 中注明，例如 `Closes #12` 或 `BREAKING CHANGE: ...`。

推荐：

```text
Fix(tensor): Reject mismatched storage and shape

Validate the element count when constructing a tensor so invalid layouts
cannot enter the operator layer.

Closes #12
```

不推荐：

```text
fix bug
tensor update.
Chapter 1.1: tensor
```

## 2. Main 分支提交

课程成果和较大变更在功能分支上开发，再通过本地 `git merge --squash` 合入 `main`。小型文档、配置或明确的独立修复可以直接提交到 `main`，但必须使用仓库级标签。

禁止把 `Add(scope): ...`、`Wip(scope): ...` 等开发过程提交直接合并到 `main`。

### 2.1 课程提交

属于课程主线的成果使用：

```text
Chapter<Stage>.<Chapter>.<Sequence>: <Subject>
```

编号含义：

- `Stage`：教程中的阶段号。
- `Chapter`：该阶段内的章节号。
- `Sequence`：该章节在 `main` 上的成果序号，从 `1` 开始递增。

例如，教程中的 `Chapter 1.1 Storage + Shape` 可以产生：

```text
Chapter1.1.1: Implement Tensor storage and shape
Chapter1.1.2: Add N-dimensional indexing
Chapter1.1.3: Validate Tensor construction invariants
```

规则：

1. `Chapter` 与编号之间不加空格。
2. 三段编号都必须存在，只使用十进制数字。
3. 同一章节的 `Sequence` 必须相对 `main` 上一个编号递增；准备合并前再次确认，避免并行分支使用相同编号。
4. 只要变更明确属于某个课程章节，即使内容主要是测试、修复或重构，也优先使用 `Chapter...`，不要改用仓库级标签。
5. 一个 squash 提交如果横跨多个章节，应优先拆分；确实无法拆分时，以主要交付物所属章节编号命名，并在 commit body 中列出其余影响。

课程提交标题应匹配：

```regex
^Chapter[0-9]+\.[0-9]+\.[0-9]+: [A-Z].+$
```

### 2.2 仓库级提交

无法归属于单一课程章节的变更使用以下标签：

| Label | 用途 | 示例 |
| --- | --- | --- |
| `Docs` | 跨章节文档、贡献指南 | `Docs: Add commit message convention` |
| `Fix` | 跨章节或基础设施缺陷修复 | `Fix: Correct reference data generator paths` |
| `Refactor` | 不改变行为的跨章节重构 | `Refactor: Unify shared test helpers` |
| `Test` | 跨章节测试设施或测试数据 | `Test: Add numerical tolerance helpers` |
| `Bench` | 基准测试基础设施 | `Bench: Record CPU benchmark environment` |
| `Perf` | 跨章节性能优化 | `Perf: Reduce reference data loading overhead` |
| `Build` | 构建、依赖和打包配置 | `Build: Pin development dependencies` |
| `CI` | 持续集成配置 | `CI: Run Rust checks before merging` |
| `Chore` | 不属于以上类别的仓库维护 | `Chore: Update repository metadata` |
| `Revert` | 撤销一个已经进入主线的提交 | `Revert: Remove unsupported mmap fallback` |

不要为了规避课程编号而选择仓库级标签。新算子、新模型或 runtime 能力通常都是课程成果，应使用 `Chapter...`。

仓库级提交标题应匹配：

```regex
^(Docs|Fix|Refactor|Test|Bench|Perf|Build|CI|Chore|Revert): [A-Z].+$
```

## 3. 功能分支开发提交

### 3.1 分支命名

```text
chapter/<stage>.<chapter>-<short-description>
<kind>/<short-description>
```

`kind` 可以是 `fix`、`docs`、`refactor`、`test`、`bench`、`perf`、`build`、`ci`、`chore` 或 `revert`。

示例：

```text
chapter/1.1-tensor-storage
chapter/3.6-single-head-attention
fix/reference-shape-validation
docs/commit-convention
```

分支名使用小写英文和连字符；课程分支只写教程的两段章节号，第三段成果序号留给 squash 后的主线提交。

### 3.2 过程提交格式

功能分支中的提交使用：

```text
<Action>(<scope>): <Subject>
```

`scope` 必填，使用小写英文，表示受影响的模块或主题，例如 `tensor`、`matmul`、`runtime`、`docs` 或 `ci`。

| Action | 用途 |
| --- | --- |
| `Add` | 增加新的可用行为 |
| `Fix` | 修复当前分支或已有代码中的缺陷 |
| `Test` | 增加或调整测试 |
| `Refactor` | 调整结构且不改变外部行为 |
| `Docs` | 更新文档或注释 |
| `Bench` | 增加或调整性能基准 |
| `Perf` | 优化性能且不改变预期行为 |
| `Build` | 调整依赖或构建配置 |
| `CI` | 调整持续集成 |
| `Chore` | 其他维护工作 |
| `Spike` | 记录实验性实现；合并前允许被后续提交替代 |
| `Wip` | 临时检查点；仅在确实需要共享未完成状态时使用 |

推荐的开发过程：

```text
Test(tensor): Cover valid and invalid shapes
Add(tensor): Implement checked Tensor constructor
Refactor(tensor): Extract shape element counting
Fix(tensor): Handle zero-sized dimensions
Docs(tensor): Document constructor invariants
```

这些提交应让开发期间的历史能够按顺序反映设计如何形成。除 `Spike` 和 `Wip` 外，每个提交都应能够编译并通过当时已有的相关测试。`Wip` 不能作为 squash 后的主线提交。

开发分支提交标题应匹配：

```regex
^(Add|Fix|Test|Refactor|Docs|Bench|Perf|Build|CI|Chore|Spike|Wip)\([a-z0-9][a-z0-9-]*\): [A-Z].+$
```

## 4. 本地 Squash 流程

整个流程只依赖 Git，不要求 GitHub、Pull Request 或其他托管平台。

从最新的 `main` 创建课程分支：

```bash
git switch main
git switch -c chapter/1.1-tensor-storage
```

开发期间按过程提交格式提交：

```bash
git commit -m "Test(tensor): Cover valid and invalid shapes"
git commit -m "Add(tensor): Implement checked Tensor constructor"
git commit -m "Fix(tensor): Handle zero-sized dimensions"
```

完成开发并通过测试后，在本地 squash 到 `main`：

```bash
git switch main
git merge --squash chapter/1.1-tensor-storage
git commit -m "Chapter1.1.1: Implement Tensor storage and shape"
```

`git merge --squash` 会把分支的最终改动放入暂存区，但不会把过程提交带入 `main`。提交前应检查暂存内容、确认课程成果序号并运行相关测试。

确认主线提交正确后，可以删除不再需要的功能分支。由于 squash 不会建立普通 merge 的祖先关系，删除前应确认分支中没有遗漏的改动。

对于无需功能分支的小型仓库维护，可以直接在 `main` 提交：

```bash
git commit -m "Docs: Update commit message examples"
```

## 5. 自动检查

提交规则统一实现在 `scripts/validate-commit-message.sh` 中，本地 hook 和 CI 共用该脚本，避免两套正则产生差异。检查内容包括：

- 标题不超过 72 个字符。
- 在 `main` 上使用 `Chapter...` 或仓库级标签格式。
- 在其他分支上使用 `Action(scope): Subject` 格式。
- `Subject` 以英文大写字母开头。

### 5.1 本地检查

仓库通过 `.githooks/commit-msg` 在执行 `git commit` 时调用公共检查脚本。

Git 不会随仓库自动启用自定义 hook。**每次重新 clone 仓库后，都必须在仓库根目录执行一次：**

```bash
git config core.hooksPath .githooks
```

可以用下面的命令确认配置已经生效：

```bash
git config --get core.hooksPath
```

预期输出为：

```text
.githooks
```

该配置只写入当前仓库的 `.git/config`，不会影响其他仓库，也不会被提交到 Git 历史。hook 脚本本身会随仓库提交和更新。

特殊情况下可以通过 `git commit --no-verify` 跳过检查，但不应将它作为日常提交方式。

### 5.2 CI 检查

`.github/workflows/ci.yml` 在每次 push 到 `main` 后执行：

1. 检查本次 push 新增的所有 commit message。
2. 在 Ubuntu 24.04、Windows Server 2025 和 macOS 15 上分别输出 Rust 与 Cargo 版本。
3. 在三个平台上分别执行 `cargo fmt --all -- --check`。
4. 在三个平台上分别执行严格的 Clippy 检查。
5. 在三个平台上分别执行所有测试。

macOS 15 runner 使用 Apple Silicon，因此该矩阵同时覆盖 x64 Linux、x64 Windows 和 ARM64 macOS。矩阵设置 `fail-fast: false`，单个平台失败时不会取消其他平台的任务。

也可以在 GitHub Actions 页面通过 `workflow_dispatch` 手动运行。手动运行没有对应的新增提交范围，因此只执行 Rust 检查。

CI 在 push 完成后才会启动，所以它只能报告已经进入远端 `main` 的错误提交，不能代替本地 hook。为了确保文档或配置提交也接受命名检查，CI 不设置文件路径过滤。

## 6. 完整示例

开发 `Chapter 3.6 Single-Head Attention` 时：

```text
Branch: chapter/3.6-single-head-attention

Test(attention): Add masked attention reference cases
Add(attention): Compute scaled dot-product scores
Add(attention): Apply causal attention mask
Fix(attention): Stabilize softmax for large logits
Refactor(attention): Reuse batched matmul helpers
```

本地 squash 后的 `main` 提交：

```text
Chapter3.6.1: Implement single-head causal attention
```

这样，功能分支上的提交服务于开发过程，`main` 的历史只体现最终的课程里程碑。本地分支和 squash 流程不依赖托管平台；GitHub CI 只作为额外的远端验证。
