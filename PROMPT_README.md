# 矛盾条件生成 - Prompt 文件说明

## 📁 文件清单

| 文件名 | 步骤 | 使用模型 | 用途 |
|--------|------|----------|------|
| `extract.txt` | Step 1 | GPT-4o-mini | 提取问题中的所有关键条件 |
| `contradict_analysis.txt` | Step 2.1 | DeepSeek-R1 | 分析如何为某个条件添加矛盾 |
| `contradict_rewrite.txt` | Step 2.2 | DeepSeek-R1 | 生成添加矛盾后的问题 |
| `contradict_verify_s1.txt` | Step 3.1 | GPT-4o | 验证是否只修改了一个条件 |
| `contradict_verify_s2.txt` | Step 3.2 | DeepSeek-V3 | 提取矛盾条件的描述 |
| `contradict_unsolve_s1.txt` | Step 3.3 | DeepSeek-R1 | 分析为什么问题不可解 |
| `contradict_unsolve_s2.txt` | Step 3.4 | DeepSeek-R1 | 判断是否真的不可解 |
| `contradict_unsolve_s3.txt` | Step 3.5 | DeepSeek-V3 | 提取简洁的不可解原因 |

---

## 🔄 Pipeline 流程

```
原始问题
    ↓
[Step 1] extract.txt
    → 提取所有条件: ["条件1", "条件2", "条件3"]
    ↓
[Step 2] 对每个条件:
    ├─ [2.1] contradict_analysis.txt
    │   → 分析如何添加矛盾
    └─ [2.2] contradict_rewrite.txt
        → 生成矛盾问题
    ↓
[Step 3] 验证每个矛盾问题:
    ├─ [3.1] contradict_verify_s1.txt
    │   → 验证单条件修改 (True/False)
    ├─ [3.2] contradict_verify_s2.txt
    │   → 提取矛盾描述
    ├─ [3.3] contradict_unsolve_s1.txt
    │   → 分析不可解性
    ├─ [3.4] contradict_unsolve_s2.txt
    │   → 判断是否真的不可解 (True/False)
    └─ [3.5] contradict_unsolve_s3.txt
        → 提取不可解原因
    ↓
有效的矛盾问题
```

---

## 📝 详细说明

### Step 1: extract.txt

**输入变量**:
- `{original_math_question}`: 原始数学问题

**输出格式**: JSON数组
```json
[
  "条件1: z是复数，|z| = 4",
  "条件2: w是复数，|w| = 3",
  "条件3: z + w的实部最大"
]
```

**模型**: GPT-4o-mini (快速、便宜)

---

### Step 2.1: contradict_analysis.txt

**输入变量**:
- `{original_math_question}`: 原始问题
- `{original_answer}`: 原始答案
- `{extracted_condition}`: 某个关键条件

**输出格式**: 自然语言分析（2-5句话）

**示例输出**:
```
如果删除 "|z| = 4" 这个约束，z的模可以任意大。
当 |z| → ∞ 时，表达式 (75+117i)z 的实部将趋向无穷大，
因此无法找到"最大的实部"，问题变得不可解。
```

**模型**: DeepSeek-R1 (深度推理)

---

### Step 2.2: contradict_rewrite.txt

**输入变量**:
- `{original_math_question}`: 原始问题
- `{original_answer}`: 原始答案
- `{extracted_condition}`: 要矛盾的条件

**输出格式**: 重写后的数学问题

**示例输出**:
```
Find the largest possible real part of
[(75+117i)z + (96+144i)/z]
where z is a complex number.
```

**注意**: 删除了 "|z| = 4" 约束

**模型**: DeepSeek-R1 (创造性重写)

---

### Step 3.1: contradict_verify_s1.txt

**输入变量**:
- `{original_question}`: 原始问题
- `{rewritten_question}`: 重写后的问题

**输出格式**: 单词 "True" 或 "False"

**判断标准**:
- True: 只修改了一个条件
- False: 修改了多个条件或其他改动

**模型**: GPT-4o (高准确度判断)

---

### Step 3.2: contradict_verify_s2.txt

**输入变量**:
- `{original_question}`: 原始问题
- `{original_condition}`: 原始条件
- `{rewritten_question}`: 重写后的问题

**输出格式**: 简短描述

**示例输出**:
```
Delete |z| = 4 constraint
```

**模型**: DeepSeek-V3 (快速提取)

---

### Step 3.3: contradict_unsolve_s1.txt

**输入变量**:
- `{original_question}`: 原始问题
- `{original_answer}`: 原始答案
- `{rewritten_question}`: 重写后的问题

**输出格式**: 详细分析（2-5句话）

**示例输出**:
```
Without the constraint |z|=4, the variable z can have any
modulus. As |z| approaches infinity, the term (75+117i)z
dominates and its real part grows without bound. Therefore,
there is no finite maximum for the real part, making the
problem unsolvable.
```

**模型**: DeepSeek-R1 (深度分析)

---

### Step 3.4: contradict_unsolve_s2.txt

**输入变量**:
- `{original_question}`: 原始问题
- `{original_answer}`: 原始答案
- `{rewritten_question}`: 重写后的问题
- `{unsolvability_analysis}`: Step 3.3 的分析

**输出格式**: 单词 "True" 或 "False"

**判断标准**:
- True: 真的不可解/无唯一解
- False: 仍然可解

**模型**: DeepSeek-R1 (推理判断)

---

### Step 3.5: contradict_unsolve_s3.txt

**输入变量**:
- `{original_question}`: 原始问题
- `{rewritten_question}`: 重写后的问题
- `{unsolvability_analysis}`: Step 3.3 的分析

**输出格式**: 简洁原因（1-2句话）

**示例输出**:
```
The constraint |z|=4 is essential to bound the real part.
Without it, the expression is unbounded above.
```

**模型**: DeepSeek-V3 (提取总结)

---

## 🎯 设计原则

### 1. 模型分工明确

- **GPT-4o-mini**: 简单提取任务（成本低）
- **GPT-4o**: 高准确度判断（验证关键步骤）
- **DeepSeek-R1**: 深度推理和分析（性价比高）
- **DeepSeek-V3**: 快速提取和总结（速度快）

### 2. 多重验证机制

每个矛盾问题需要通过：
1. ✅ 单条件修改验证
2. ✅ 矛盾描述可提取
3. ✅ 不可解性分析合理
4. ✅ 真的不可解

### 3. 可定制性

所有prompt都可以根据需要修改：
- 添加更多示例
- 调整输出格式
- 修改判断标准

---

## 📊 质量控制

### 自动过滤点

| 步骤 | 过滤条件 | 失败原因标记 |
|-----|---------|-------------|
| 2.1 | 分析长度 < 10 | 跳过该条件 |
| 2.2 | 问题长度 < 20 | 跳过该条件 |
| 3.1 | 验证 = False | `multiple_conditions_changed` |
| 3.2 | 描述长度 < 5 | `no_contradicted_condition` |
| 3.4 | 判断 = False | `still_solvable` |

### 预期通过率

根据论文数据估算：
- Step 1 → Step 2: ~70-80% 的条件能生成矛盾
- Step 2 → Step 3: ~85-90% 通过验证
- **总体通过率**: ~60-70%

---

## 🔧 自定义修改

### 添加示例（推荐）

在每个prompt的末尾添加 few-shot 示例可以提高质量：

```txt
### Examples ###:

Example 1:
Original condition: "x > 0"
Contradicted: "x < 0"

Example 2:
Original condition: "|z| = 4"
Contradicted: "Remove |z| = 4 constraint"
```

### 调整输出格式

如果需要结构化输出，可以要求JSON格式：

```txt
### Output Format ###:
{
  "contradicted_condition": "...",
  "reason": "..."
}
```

### 修改判断标准

可以在 `contradict_verify_s1.txt` 中放宽/严格判断：

```txt
# 严格模式
判断标准: EXACTLY one condition changed, NOTHING else

# 宽松模式
判断标准: At most one condition changed significantly
```

---

## 📖 使用建议

1. **先用小数据集测试**
   ```bash
   python code/contradiction_construction.py --dataset test --test_mode
   ```

2. **检查中间输出**
   - 查看 `_contradictions.json` 中生成的问题是否合理
   - 查看 `_final.json` 中的失败原因分布

3. **迭代优化prompt**
   - 如果通过率低，在prompt中添加更多示例
   - 如果质量差，加强判断标准

4. **成本优化**
   - 可以用更便宜的模型替换（如全用GPT-4o-mini）
   - 可以减少验证步骤（风险：质量下降）

---

## 📚 参考文献

- ReliableMath论文: Table 4 (Contradiction 关键词分析)
- 原始使用指南: 你提供的中文文档
- 代码实现: `code/contradiction_construction.py`

---

**最后更新**: 2025-11-18
**维护者**: Claude Code
