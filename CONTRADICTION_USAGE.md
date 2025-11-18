# 矛盾条件生成 - 使用文档

## 📖 概述

基于您提供的"缺省条件"代码架构，已完成"矛盾条件"模块的实现。

### ✅ 已完成内容

1. **5个prompt文件** (`prompt/v4-comp/rewrite/`)
   - ✅ `contradict_verify_s1.txt` - 验证单条件修改
   - ✅ `contradict_verify_s2.txt` - 提取矛盾描述
   - ✅ `contradict_unsolve_s1.txt` - 分析不可解性
   - ✅ `contradict_unsolve_s2.txt` - 判断是否真的不可解
   - ✅ `contradict_unsolve_s3.txt` - 提取不可解原因

2. **主程序** (`code/contradiction_construction.py`)
   - ✅ 完整的3步pipeline
   - ✅ 并行处理支持
   - ✅ 断点续传
   - ✅ Token使用统计

---

## 🏗️ 架构对比

### 缺省条件 vs 矛盾条件

| 特性 | 缺省条件 (Removal) | 矛盾条件 (Contradiction) |
|------|-------------------|-------------------------|
| **操作** | 删除条件 | 添加矛盾条件 |
| **组合数** | C(N,n) 种组合 | N 个独立变体 |
| **验证方式** | 两轮验证（必要性+充分性） | 单条件验证+不可解性分析 |
| **复杂度** | 需要vLLM采样验证 | 需要深度推理分析 |

---

## 💻 使用方法

### 1. 准备工作

确保 `data/api_keys.json` 包含以下模型配置：

```json
{
  "gpt-4o-mini": [["gpt-4o-mini", "sk-xxx", "https://api.openai.com/v1"]],
  "gpt-4o": [["gpt-4o", "sk-xxx", "https://api.openai.com/v1"]],
  "deepseek-r1": [["deepseek-reasoner", "sk-xxx", "https://api.deepseek.com"]],
  "deepseek-v3": [["deepseek-chat", "sk-xxx", "https://api.deepseek.com"]]
}
```

### 2. 准备输入数据

输入数据格式（与缺省条件相同）：

```json
[
  {
    "id": 1,
    "data_source": "AIME",
    "difficulty": "hard",
    "question": "原始数学问题...",
    "ground_truth": "答案"
  }
]
```

放置在 `data/solve/your_dataset.json`

### 3. 运行生成

```bash
cd ~/ReliableMath

# 基础运行
python code/contradiction_construction.py --dataset aime

# 测试模式（只处理前5条）
python code/contradiction_construction.py --dataset aime --test_mode

# 强制重新处理
python code/contradiction_construction.py --dataset aime --force

# 自定义模型配置
python code/contradiction_construction.py \
  --dataset aime \
  --model gpt-4o-mini \
  --analysis_model deepseek-r1 \
  --verify_model gpt-4o \
  --extract_model deepseek-v3

# 自定义并行线程数
python code/contradiction_construction.py --dataset aime --threads 16
```

### 4. 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--dataset` | `aime` | 数据集名称 |
| `--model` | `gpt-4o-mini` | 条件提取模型 |
| `--analysis_model` | `deepseek-r1` | 分析模型（DeepSeek-R1） |
| `--verify_model` | `gpt-4o` | 验证模型（GPT-4o） |
| `--extract_model` | `deepseek-v3` | 提取模型（DeepSeek-V3） |
| `--data_dir` | `data/solve` | 输入目录 |
| `--output_dir` | `data/construct_contradiction` | 输出目录 |
| `--prompt_dir` | `prompt/v4-comp/rewrite` | Prompt目录 |
| `--temperature` | `0.0` | 生成温度 |
| `--threads` | `8` | 并行线程数 |
| `--test_mode` | `False` | 测试模式（只处理前5条） |
| `--force` | `False` | 强制重新处理 |

---

## 📊 输出格式

### 中间文件

1. **`{dataset}_conditions.json`** - Step 1输出
   ```json
   {
     "id": 1,
     "question": "...",
     "extracted_condition": ["条件1", "条件2", "条件3"],
     "num_conditions": 3
   }
   ```

2. **`{dataset}_contradictions.json`** - Step 2输出
   ```json
   {
     "id": 1,
     "contradiction_variants": [
       {
         "variant_id": "1_contradict_0",
         "extracted_condition": "原始条件",
         "analysis": "如何添加矛盾的分析...",
         "contradicted_question": "添加矛盾后的问题..."
       }
     ]
   }
   ```

3. **`{dataset}_final.json`** - Step 3输出
   - 包含完整的验证信息

### 最终输出

**`{dataset}_valid.json`** - 只包含通过验证的矛盾问题

```json
[
  {
    "id": "1_contradict_0",
    "data_source": "AIME",
    "difficulty": "hard",
    "transformation_type": "contradiction",
    "original_question": "原始问题...",
    "ground_truth": "答案",
    "extracted_condition": "z is a complex number with |z| = 4",
    "contradict_question": "重写后的矛盾问题...",
    "rewritten_condition": "Delete |z| = 4 constraint",
    "unsolvable_reason": "Without the constraint |z|=4, the expression is unbounded.",
    "verification": {
      "single_condition_verified": true,
      "is_truly_unsolvable": true,
      "is_valid": true
    },
    "original_id": 1
  }
]
```

---

## 🔧 Pipeline详解

### Step 1: 提取条件 (extract_conditions_only)

**使用模型**: `gpt-4o-mini`

**Prompt**: `extract.txt`

**输出**:
```python
data["extracted_condition"] = ["条件1", "条件2", ...]
data["num_conditions"] = N
```

---

### Step 2: 生成矛盾变体 (generate_contradiction_variants)

**使用模型**: `deepseek-r1`

对每个条件执行：

#### 2.1 分析如何添加矛盾
**Prompt**: `contradict_analysis.txt`
- 输入: 原问题 + 答案 + 条件
- 输出: 如何添加矛盾的分析

#### 2.2 生成矛盾问题
**Prompt**: `contradict_rewrite.txt`
- 输入: 原问题 + 答案 + 条件
- 输出: 添加矛盾后的问题

**过滤条件**:
- ✗ 分析为空 → 跳过
- ✗ 重写问题太短 → 跳过

---

### Step 3: 验证矛盾条件 (verify_contradiction_validity)

对每个variant执行以下验证：

#### 3.1 验证单条件修改
**模型**: `gpt-4o`
**Prompt**: `contradict_verify_s1.txt`
- 判断是否只修改了一个条件
- 返回: True/False
- ✗ False → 标记为invalid，跳过后续步骤

#### 3.2 提取矛盾描述
**模型**: `deepseek-v3`
**Prompt**: `contradict_verify_s2.txt`
- 提取清晰的矛盾条件描述
- ✗ 提取失败 → 标记为invalid

#### 3.3 分析不可解性
**模型**: `deepseek-r1`
**Prompt**: `contradict_unsolve_s1.txt`
- 深度分析为什么问题不可解
- 输出: 详细分析（2-5句话）

#### 3.4 判断是否真的不可解
**模型**: `deepseek-r1`
**Prompt**: `contradict_unsolve_s2.txt`
- 基于分析判断问题是否真的不可解
- 返回: True/False
- ✗ False → 标记为invalid

#### 3.5 提取不可解原因
**模型**: `deepseek-v3`
**Prompt**: `contradict_unsolve_s3.txt`
- 提取简洁的不可解原因（1-2句话）
- 用于最终数据集

**最终判定**:
```python
is_valid = (单条件验证 ✓) AND (矛盾描述提取 ✓) AND (真的不可解 ✓)
```

---

## 📈 质量控制

### 自动过滤规则

1. **Step 2 过滤**:
   - 分析长度 < 10字符 → 跳过
   - 重写问题长度 < 20字符 → 跳过

2. **Step 3 过滤**:
   - 单条件验证失败 → `failure_reason: "multiple_conditions_changed"`
   - 矛盾描述提取失败 → `failure_reason: "no_contradicted_condition"`
   - 不可解性判断失败 → `failure_reason: "still_solvable"`

### 通过率预估

根据论文Table 2的数据：

| Dataset | 原始 | Step 1&2 生成 | Step 3 人工审核后 | 通过率 |
|---------|------|--------------|------------------|--------|
| AIME    | 30   | 71           | 65               | ~92% |
| MATH    | 100  | 216          | 164              | ~76% |

预期自动验证通过率: **70-85%**

---

## 🔍 调试与监控

### 查看处理进度

```bash
# 查看条件提取进度
ls -lh data/construct_contradiction/aime_conditions.json*

# 查看矛盾生成进度
ls -lh data/construct_contradiction/aime_contradictions.json*

# 查看验证进度
ls -lh data/construct_contradiction/aime_final.json*
```

### 断点续传机制

代码会自动检测 `.jsonl` 中间文件：
- ✅ 已处理的ID会自动跳过
- ✅ 支持Ctrl+C中断后继续
- ✅ 使用 `--force` 强制重新处理

### 日志示例

```
[1/3] Extracting conditions (parallel)
Extracting conditions: 100%|████████| 30/30 [00:15<00:00,  1.95it/s]

[2/3] Generating contradictions (parallel)
ID 77: Generating contradictions for 3 conditions
ID 77_contradict_0: ✓ Generated contradiction
ID 77_contradict_1: ✓ Generated contradiction
Generating contradictions: 100%|████████| 30/30 [01:23<00:00,  2.78s/it]

[3/3] Verifying contradictions (parallel)
ID 77_contradict_0: Starting verification...
ID 77_contradict_0: ✓ Single condition verified
ID 77_contradict_0: ✓ Contradicted condition extracted
ID 77_contradict_0: ✓ Confirmed unsolvable
ID 77_contradict_0: 🎉 VALID - All checks passed!
Verifying contradictions: 100%|████████| 30/30 [02:15<00:00,  4.50s/it]
```

---

## 💰 成本估算

假设处理100道题，平均每题3个条件：

| 模型 | 用途 | 调用次数 | Token/次 | 总Token | 成本 |
|-----|------|---------|----------|---------|------|
| GPT-4o-mini | 条件提取 | 100 | 1000 | 100K | $0.015 |
| DeepSeek-R1 | 矛盾分析 | 300 | 2000 | 600K | ~$0.60 |
| DeepSeek-R1 | 矛盾重写 | 300 | 2000 | 600K | ~$0.60 |
| GPT-4o | 单条件验证 | 300 | 500 | 150K | $0.50 |
| DeepSeek-V3 | 提取描述 | 250 | 1000 | 250K | ~$0.25 |
| DeepSeek-R1 | 不可解分析 | 250 | 2000 | 500K | ~$0.50 |
| DeepSeek-R1 | 不可解判断 | 250 | 1500 | 375K | ~$0.38 |
| DeepSeek-V3 | 提取原因 | 200 | 1000 | 200K | ~$0.20 |

**预估总成本**: ~$3-4 / 100题

---

## ⚠️ 常见问题

### Q1: 为什么有些条件没有生成矛盾？

**可能原因**:
1. DeepSeek-R1认为该条件不适合添加矛盾（analysis为空）
2. 生成的矛盾问题太短/格式不对
3. 单条件验证失败（改变了多个条件）

**解决方案**:
- 检查 `_contradictions.json` 中的 `contradiction_variants` 数量
- 查看日志中的跳过原因

### Q2: 通过率太低怎么办？

**优化建议**:
1. **提高条件提取质量**: 使用更好的模型（如GPT-4o）
2. **调整Prompt**: 在prompt中添加更多示例
3. **放宽验证标准**: 修改 `verify_s1` 的判断逻辑

### Q3: 如何并行处理多个数据集？

```bash
# 使用不同终端窗口
# Terminal 1
python code/contradiction_construction.py --dataset aime

# Terminal 2
python code/contradiction_construction.py --dataset amc

# Terminal 3
python code/contradiction_construction.py --dataset math
```

### Q4: 代码与原文档的差异？

**主要改动**:
1. ✅ 保留了缺省条件代码的所有基础设施
2. ✅ 使用并行处理框架（更快）
3. ✅ 自动断点续传（更稳定）
4. ✅ 详细的Token统计（更透明）

**与原文档的兼容性**:
- ✅ 输出格式完全兼容
- ✅ 验证流程完全一致
- ✅ Prompt模板可自定义

---

## 🎯 下一步

1. **测试运行**:
   ```bash
   python code/contradiction_construction.py --dataset aime --test_mode
   ```

2. **查看结果**:
   ```bash
   cat data/construct_contradiction/aime_valid.json | jq '.[0]'
   ```

3. **人工抽样审核**:
   - 随机抽取10-20个样本
   - 检查矛盾是否合理
   - 检查不可解原因是否准确

4. **全量处理**:
   ```bash
   python code/contradiction_construction.py --dataset aime
   ```

---

## 📚 参考

- **原始文档**: 您提供的使用指南
- **论文**: ReliableMath: Benchmark of Reliable Mathematical Reasoning for LLMs
- **代码基础**: `construct_mip_with_deepscaler_num_missing.py`

---

**祝使用顺利！如有问题，欢迎反馈 📧**
