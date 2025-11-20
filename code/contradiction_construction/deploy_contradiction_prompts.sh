#!/bin/bash

#

# 脚本: 在生产环境直接生成矛盾条件的prompt文件

# 目标: /data2/yiqianzhang/ReliableMath/prompt/contradict_data

#

 

# 目标目录

TARGET_DIR="/data2/yiqianzhang/ReliableMath/prompt/contradict_data"

 

echo "=========================================="

echo "矛盾条件 Prompt 文件生成脚本"

echo "=========================================="

echo "目标目录: $TARGET_DIR"

echo ""

 

# 创建目标目录

echo "[1/9] 创建目标目录..."

mkdir -p "$TARGET_DIR"

if [ $? -eq 0 ]; then

    echo "✓ 目录创建成功"

else

    echo "✗ 目录创建失败！"

    exit 1

fi

 

echo ""

echo "[2/9] 生成 extract.txt..."

cat > "$TARGET_DIR/extract.txt" << 'PROMPT_EOF'

Please identify and extract all the key conditions from the following mathematical problem. Return the conditions as a JSON array.

 

**Original Mathematical Question:**

{original_math_question}

 

**Instructions:**

1. Identify all essential conditions, constraints, and given information

2. Each condition should be a complete, standalone statement

3. Number the conditions if there are multiple ones

4. Return as a JSON array of strings

 

**Output Format:**

[

  "condition 1",

  "condition 2",

  "condition 3"

]

 

**Extracted Conditions:**

PROMPT_EOF

echo "✓ extract.txt 创建成功"

 

echo ""

echo "[3/9] 生成 contradict_analysis.txt..."

cat > "$TARGET_DIR/contradict_analysis.txt" << 'PROMPT_EOF'

### Analysis ###:

 

Analyze the extracted condition: "{extracted_condition}" from the original math problem. Explain how this condition constrains the problem and what would happen if this condition were contradicted (i.e., replaced by an opposing statement). Discuss whether a contradicted condition leads to an impossible scenario, a different solution, or a valid but altered problem.

 

Original question:

{original_math_question}

Original answer:

{original_answer}

 

### Rewritten Mathematical Question ###:

 

(Include below a short candidate rewrite if relevant.)

PROMPT_EOF

echo "✓ contradict_analysis.txt 创建成功"

 

echo ""

echo "[4/9] 生成 contradict_rewrite.txt..."

cat > "$TARGET_DIR/contradict_rewrite.txt" << 'PROMPT_EOF'

### Rewritten Mathematical Question ###:

 

Using the analysis above, produce a rewritten version of the original math question where the extracted condition "{extracted_condition}" is contradicted (i.e., replaced by a logically opposing statement). The rewritten question should be a coherent standalone math problem and explicitly show the contradicted condition.

 

Original question:

{original_math_question}

Original answer:

{original_answer}

 

### Rewritten Mathematical Question ###:

PROMPT_EOF

echo "✓ contradict_rewrite.txt 创建成功"

 

echo ""

echo "[5/9] 生成 contradict_verify_s1.txt..."

cat > "$TARGET_DIR/contradict_verify_s1.txt" << 'PROMPT_EOF'

### Task ###:

 

You are given an ORIGINAL mathematical question and a REWRITTEN version. The rewritten version should have ONLY ONE condition modified (contradicted) compared to the original.

 

Your task: Verify whether the rewritten question changed ONLY ONE condition, keeping everything else identical.

 

### Original Question ###:

{original_question}

 

### Rewritten Question ###:

{rewritten_question}

 

### Instructions ###:

1. Compare the two questions carefully

2. Identify ALL differences between them

3. Determine if there is EXACTLY ONE condition that was contradicted/modified

4. Return your verdict

 

### Output Format ###:

Answer with ONLY one word:

- "True" if EXACTLY one condition was modified and everything else is identical

- "False" if multiple conditions were changed, or if the rewrite is invalid

 

### Answer ###:

PROMPT_EOF

echo "✓ contradict_verify_s1.txt 创建成功"

 

echo ""

echo "[6/9] 生成 contradict_verify_s2.txt..."

cat > "$TARGET_DIR/contradict_verify_s2.txt" << 'PROMPT_EOF'

### Task ###:

 

Extract the CONTRADICTED condition from the rewritten mathematical question.

 

### Original Question ###:

{original_question}

 

### Original Condition (that should be contradicted) ###:

{original_condition}

 

### Rewritten Question (with contradiction) ###:

{rewritten_question}

 

### Instructions ###:

Compare the original and rewritten questions. Extract a clear description of how the condition was contradicted. This should be a concise statement describing the new contradictory condition.

 

Examples:

- Original: "|z| = 4"  →  Contradicted: "Delete |z| = 4 constraint"

- Original: "ABCD on hyperbola x²/20 - y²/24 = 1"  →  Contradicted: "B, C, D on hyperbola x²/20 - y²/24 = 2"

- Original: "x > 0"  →  Contradicted: "x < 0"

 

### Output Format ###:

Provide ONLY the contradicted condition description, without any additional explanation.

 

### Contradicted Condition ###:

PROMPT_EOF

echo "✓ contradict_verify_s2.txt 创建成功"

 

echo ""

echo "[7/9] 生成 contradict_unsolve_s1.txt..."

cat > "$TARGET_DIR/contradict_unsolve_s1.txt" << 'PROMPT_EOF'

### Task ###:

 

Analyze why the rewritten mathematical question (with contradicted condition) is UNSOLVABLE or has no unique solution.

 

### Original Question ###:

{original_question}

 

### Original Answer ###:

{original_answer}

 

### Rewritten Question (with contradiction) ###:

{rewritten_question}

 

### Instructions ###:

1. Analyze the logical structure of the rewritten question

2. Identify why the contradicted condition makes the problem unsolvable

3. Explain the mathematical reasoning clearly

4. Consider different types of unsolvability:

   - Logical contradiction (impossible conditions)

   - Unbounded solution (no unique answer)

   - Empty solution set (no solutions exist)

   - Insufficient constraints (multiple valid answers)

 

### Output Format ###:

Provide a detailed analysis (2-5 sentences) explaining why this problem cannot be solved uniquely.

 

### Analysis ###:

PROMPT_EOF

echo "✓ contradict_unsolve_s1.txt 创建成功"

 

echo ""

echo "[8/9] 生成 contradict_unsolve_s2.txt..."

cat > "$TARGET_DIR/contradict_unsolve_s2.txt" << 'PROMPT_EOF'

### Task ###:

 

Based on the analysis, judge whether the rewritten question is TRULY UNSOLVABLE (has no unique solution).

 

### Original Question ###:

{original_question}

 

### Original Answer ###:

{original_answer}

 

### Rewritten Question (with contradiction) ###:

{rewritten_question}

 

### Analysis of Unsolvability ###:

{unsolvability_analysis}

 

### Instructions ###:

Based on the analysis above, make a final judgment:

- True: The rewritten question is genuinely unsolvable or has no unique solution

- False: The question can still be solved uniquely despite the modification

 

### Output Format ###:

Answer with ONLY one word: "True" or "False"

 

### Judgment ###:

PROMPT_EOF

echo "✓ contradict_unsolve_s2.txt 创建成功"

 

echo ""

echo "[9/9] 生成 contradict_unsolve_s3.txt..."

cat > "$TARGET_DIR/contradict_unsolve_s3.txt" << 'PROMPT_EOF'

### Task ###:

 

Extract a concise reason why the rewritten question is unsolvable.

 

### Original Question ###:

{original_question}

 

### Rewritten Question (with contradiction) ###:

{rewritten_question}

 

### Full Analysis ###:

{unsolvability_analysis}

 

### Instructions ###:

Based on the analysis, extract a SHORT, CLEAR reason (1-2 sentences) explaining why this problem is unsolvable.

 

This should be suitable for storing in a dataset as the "unsolvable_reason" field.

 

### Output Format ###:

Provide ONLY the concise reason, without headers or extra explanation.

 

### Unsolvable Reason ###:

PROMPT_EOF

echo "✓ contradict_unsolve_s3.txt 创建成功"

 

echo ""

echo "=========================================="

echo "生成完成！"

echo "=========================================="

echo "目标目录: $TARGET_DIR"

echo ""

echo "生成的文件列表："

ls -lh "$TARGET_DIR"

 

echo ""

echo "文件数量："

file_count=$(ls -1 "$TARGET_DIR" | wc -l)

echo "  生成: $file_count 个文件"

echo "  预期: 8 个文件"

 

if [ "$file_count" -eq 8 ]; then

    echo ""

    echo "✓ 所有文件生成成功！"

    echo ""

    echo "使用方法："

    echo "cd /data2/yiqianzhang/ReliableMath"

    echo ""

    echo "python code/contradiction_construction/contradiction_construction.py \\"

    echo "  --dataset aime \\"

    echo "  --prompt_dir $TARGET_DIR \\"

    echo "  --test_mode"

else

    echo ""

    echo "⚠ 文件数量不匹配，请检查！"

    exit 1

fi

 

echo "=========================================="

PROMPT_EOF

 

chmod +x "$TARGET_DIR/deploy_contradiction_prompts.sh"

echo "Script created successfully!"