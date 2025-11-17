# MIP 数据集构建工具 - 可变缺省条件数量版本

 

## 📋 概述

 

本工具是 MIP (Missing Information Problem) 数据集构建系统的改进版本，支持**可变数量的条件缺省**。

 

### 核心改进

 

与原版本相比，主要改进：

 

1. **分离架构**：将"条件提取"与"移除改写"分为两个独立步骤

2. **灵活控制**：通过 `--num_missing` 参数控制缺省条件数量

3. **组合生成**：自动生成所有 C(N, n) 种组合变体

 

### 流程对比

 

**原版本（单步）**：

```

Extract + Remove + Rewrite (一个 prompt)

  ↓

  只能生成 n=1 的变体

```

 

**新版本（两步）**：

```

Step 1: Extract Conditions (提取条件)

  ↓

Step 2: Remove + Rewrite (参数化缺省数量)

  ↓

Step 3-4: Two-Round Verification (保持不变)

```

 

---

 

## 🚀 快速开始

 

### 1. 环境要求

 

- Python 3.8+

- 已配置 `data/api_keys.json`

- 已安装 deepscaler 模块

 

### 2. 基础用法

 

```bash

# 确保在 ~/ReliableMath 目录下运行

cd /data2/yiqianzhang/ReliableMath

 

# 基础运行（缺省 1 个条件）

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 1

 

# 缺省 2 个条件（更高难度）

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 2

 

# 缺省 3 个条件（极高难度）

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 3

```

 

### 3. 测试模式

 

```bash

# 只处理前 5 个样本（快速测试）

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 2 \

  --test_mode

```

 

---

 

## 📊 数学原理

 

### 组合数计算

 

对于包含 N 个条件的问题，缺省 n 个条件会生成 **C(N, n)** 种变体。

 

**示例**：

 

问题 `D = {q, c1, c2, c3}`（N=3）

 

| num_missing | 组合数 | 移除的条件组合 | 保留的条件 |

|-------------|--------|----------------|-----------|

| n=1 | C(3,1)=3 | {c1}, {c2}, {c3} | {c2,c3}, {c1,c3}, {c1,c2} |

| n=2 | C(3,2)=3 | {c1,c2}, {c1,c3}, {c2,c3} | {c3}, {c2}, {c1} |

| n=3 | C(3,3)=1 | {c1,c2,c3} | {} (无条件) |

 

### 难度级别

 

- **n=1**：低难度（移除一个关键条件）

- **n=2**：中难度（移除两个关键条件）

- **n=3+**：高难度（几乎无信息）

 

---

 

## ⚙️ 命令行参数

 

### 核心参数

 

| 参数 | 默认值 | 说明 |

|------|--------|------|

| `--dataset` | `polaris_easy_20` | 数据集名称 |

| `--num_missing` | `1` | **缺省条件数量** (n) |

| `--threads` | `8` | 并行线程数 |

 

### 模型配置

 

| 参数 | 默认值 | 用途 |

|------|--------|------|

| `--model` | `gpt-4o-mini` | 条件提取/问题改写 |

| `--verify_model` | `deepseek-r1-distill-qwen-7b` | 验证求解 |

| `--judge_model` | `gpt-4o-mini` | ORM 裁判（备用） |

 

### 验证参数

 

| 参数 | 默认值 | 说明 |

|------|--------|------|

| `--temperature` | `1.0` | 验证时的温度 |

| `--max_attempts` | `8` | 每轮 sampling 次数 |

| `--use_math_orm` | `False` | 启用 LLM ORM |

 

### 路径配置

 

| 参数 | 默认值 |

|------|--------|

| `--data_dir` | `data/solve` |

| `--output_dir` | `data/construct_mip_data` |

| `--prompt_dir` | `prompt/construct_mip_with_deepscaler_num_missing` |

 

### 控制参数

 

| 参数 | 说明 |

|------|------|

| `--test_mode` | 只处理前 5 个样本 |

| `--force` | 强制重处理所有数据 |

 

---

 

## 📁 输出文件

 

运行后会生成以下文件（假设 `--dataset=polaris_easy_20 --num_missing=2`）：

 

```

data/construct_mip_data/

├── polaris_easy_20_conditions.json          # Step 1: 提取的条件

├── polaris_easy_20_variants_n2.json         # Step 2: 生成的变体 (n=2)

├── polaris_easy_20_final_n2.json            # Step 3-4: 验证结果

└── polaris_easy_20_valid_n2.json            # 最终有效数据 ⭐

```

 

### 文件说明

 

1. **`*_conditions.json`**：包含提取的条件

   ```json

   {

     "id": 1,

     "question": "原始问题",

     "extracted_conditions": ["c1", "c2", "c3"],

     "num_conditions": 3

   }

   ```

 

2. **`*_variants_n2.json`**：包含生成的变体

   ```json

   {

     "removal_variants": [

       {

         "variant_id": "1_remove_0",

         "removed_conditions": ["c1", "c2"],

         "remaining_conditions": ["c3"],

         "incomplete_question": "改写后的问题"

       }

     ]

   }

   ```

 

3. **`*_valid_n2.json`**：最终有效数据（⭐ 主要使用这个）

   ```json

   {

     "id": "1_remove_0",

     "num_missing": 2,

     "original_question": "...",

     "incomplete_question": "...",

     "removed_conditions": ["c1", "c2"],

     "verification": {

       "is_valid": true,

       "round_a_passed": true,

       "round_b_passed": true

     }

   }

   ```

 

---

 

## 🔄 工作流程详解

 

### Step 1: 提取条件

 

**Prompt**：`extract_conditions.txt`

 

**输入**：原始问题

```

"Jason bought 1 pencil, Mike bought 2 pencils. How many pencils did Jason buy?"

```

 

**输出**：条件列表

```json

[

  "Jason bought 1 pencil",

  "Mike bought 2 pencils"

]

```

 

### Step 2: 生成变体

 

**Prompt**：`rewrite_without_conditions.txt`

 

**输入**：

- 原始问题

- 所有条件

- 要移除的条件

- 要保留的条件

 

**输出**：改写后的问题

 

**示例（num_missing=2）**：

```

移除: ["Jason bought 1 pencil", "Mike bought 2 pencils"]

保留: []

→ 改写: "How many pencils did Jason buy?"

```

 

### Step 3: 验证 A（必要性）

 

**Prompt**：`verify_without_condition.txt`

 

**测试**：缺省条件下是否不可解

- 8 次 sampling

- 全都 ≠ ground_truth → ✅ 通过

 

### Step 4: 验证 B（充分性）

 

**Prompt**：`verify_with_condition.txt`

 

**测试**：加上条件后是否可解

- 8 次 sampling

- 至少 1 个 = ground_truth → ✅ 通过

 

---

 

## 💡 使用示例

 

### 示例 1：生成不同难度级别

 

```bash

# 批量生成 n=1,2,3 的数据集

for n in 1 2 3; do

  python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

    --dataset polaris_easy_20 \

    --num_missing $n \

    --threads 8

done

```

 

生成结果：

```

data/construct_mip_data/

├── polaris_easy_20_valid_n1.json  # 低难度

├── polaris_easy_20_valid_n2.json  # 中难度

└── polaris_easy_20_valid_n3.json  # 高难度

```

 

### 示例 2：启用 ORM（更高准确率）

 

```bash

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 2 \

  --use_math_orm \

  --judge_model gpt-4o-mini

```

 

### 示例 3：快速测试

 

```bash

# 只测试前 5 个样本，缺省 2 个条件

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 2 \

  --test_mode \

  --threads 2

```

 

### 示例 4：强制重处理

 

```bash

# 清空中间文件，重新处理

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \

  --dataset polaris_easy_20 \

  --num_missing 2 \

  --force

```

 

---

 

## 📈 统计报告

 

运行完成后会输出详细统计：

 

```

======================================================================

MISSING INFORMATION PROBLEM (MIP) DATASET STATISTICS

======================================================================

Configuration: num_missing = 2

Original problems: 100

 

Total removal variants generated: 300  (100个问题 × 平均3个变体)

 

📊 Two-Round Verification Results:

  Round A passed (without conditions → can't solve): 240 (80.0%)

  Round B passed (with conditions → can solve): 210 (70.0%)

  Both rounds passed (VALID): 180 (60.0%)

 

Valid removal variants: 180

 

Round B Success Distribution (when valid):

  Candidate 1: 90 variants (50.0%)

  Candidate 2: 45 variants (25.0%)

  Candidate 3: 30 variants (16.7%)

  ...

 

Judge Method Distribution (Round B success):

  Heuristic: 150 (83.3%)

  Orm: 30 (16.7%)

 

💰 GPT-4o Token Usage:

  Prompt: 1,234,567

  Completion: 567,890

  Cost = $3.45

 

💰 GPT-4o-mini Token Usage:

  Prompt: 234,567

  Completion: 123,456

  Cost = $0.12

 

🖥️  Local Model Token Usage:

  Prompt: 5,678,901

  Completion: 3,456,789

 

🎯 Heuristic Checks (free):

  Total heuristic validations: 1,440

======================================================================

```

 

---

 

## 🛠️ 故障排除

 

### 问题 1：文件路径错误

 

**错误信息**：

```

Input not found: data/solve/polaris_easy_20.json

```

 

**解决方案**：

```bash

# 确保在正确的目录运行

cd /data2/yiqianzhang/ReliableMath

pwd  # 应该显示 /data2/yiqianzhang/ReliableMath

```

 

### 问题 2：Prompt 文件缺失

 

**错误信息**：

```

Prompt file not found: prompt/construct_mip_with_deepscaler_num_missing/extract_conditions.txt

```

 

**解决方案**：

```bash

# 检查 prompt 文件是否存在

ls -la prompt/construct_mip_with_deepscaler_num_missing/

 

# 应该包含：

# - extract_conditions.txt

# - rewrite_without_conditions.txt

# - verify_without_condition.txt

# - verify_with_condition.txt

```

 

### 问题 3：API 密钥未配置

 

**错误信息**：

```

api_keys.json not found at data/api_keys.json!

```

 

**解决方案**：

```bash

# 检查 API 配置文件

cat data/api_keys.json

 

# 格式应为：

# {

#   "gpt-4o-mini": [[model_name, api_key, base_url], ...],

#   "deepseek-r1-distill-qwen-7b": [...]

# }

```

 

### 问题 4：num_missing 过大

 

**警告信息**：

```

ID 123: num_missing=5 > N=3, skipping

```

 

**解决方案**：

- 检查数据集中问题的条件数量

- 降低 `--num_missing` 参数值

- 查看 `*_conditions.json` 中的 `num_conditions` 字段

 

---

 

## 🔬 高级用法

 

### 自定义 Prompt

 

如需修改提取/改写逻辑，编辑以下文件：

 

```bash

# 修改条件提取逻辑

vim prompt/construct_mip_with_deepscaler_num_missing/extract_conditions.txt

 

# 修改问题改写逻辑

vim prompt/construct_mip_with_deepscaler_num_missing/rewrite_without_conditions.txt

```

 

### 调整并行度

 

```bash

# 低配机器（4核）

--threads 2

 

# 高配机器（32核）

--threads 16

 

# GPU 服务器（控制对 vLLM 的并发）

--threads 4  # 避免过度并发

```

 

### 调整验证严格度

 

```bash

# 更严格（16次sampling）

--max_attempts 16

 

# 更宽松（4次sampling）

--max_attempts 4

```

 

---

 

## 📝 与原版本的对比

 

| 特性 | 原版本 | 新版本 |

|------|--------|--------|

| 条件提取 | 一步完成 | 独立步骤 |

| 缺省数量 | 固定 n=1 | 可变 n |

| 变体数量 | N 个 | C(N,n) 个 |

| 难度级别 | 单一 | 多级 |

| 重复使用 | 需重新提取 | 提取一次，多次使用 |

| 输出文件 | `*_valid.json` | `*_valid_n{n}.json` |

 

---

 

## 🎯 最佳实践

 

1. **先测试后批量**：使用 `--test_mode` 测试小样本

2. **按需启用 ORM**：默认关闭，只在需要高准确率时启用

3. **合理设置线程数**：避免过度并发导致 API 限流

4. **定期备份**：保存中间文件（`*_conditions.json`）

5. **批量生成**：一次性生成多个难度级别

 

---

 

## 📚 相关文件

 

- **主程序**：`code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py`

- **Prompt 目录**：`prompt/construct_mip_with_deepscaler_num_missing/`

- **输出目录**：`data/construct_mip_data/`

- **使用文档**：`code/construct_mip_data/README_NUM_MISSING.md`（本文件）

 

---

 

## 🆘 获取帮助

 

```bash

# 查看所有参数

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py --help

```

 

遇到问题请检查：

1. 工作目录是否正确（`~/ReliableMath`）

2. API 配置是否正确（`data/api_keys.json`）

3. Prompt 文件是否完整

4. 日志输出中的错误信息

 

---

 

**Happy MIP Construction! 🚀**