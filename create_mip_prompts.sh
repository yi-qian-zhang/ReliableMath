#!/bin/bash
# 创建 MIP Construction Prompts

PROMPT_DIR="/data/home/zyq/ReliableMath/prompt/construct_mip_data"

# 创建目录
mkdir -p "$PROMPT_DIR"

echo "Creating prompt files in $PROMPT_DIR"

# ============= 1. solve_problem.txt =============
cat > "$PROMPT_DIR/solve_problem.txt" << 'EOF'
Solve the following mathematical problem and provide only the final answer.

Problem: {question}

**Instructions**:
1. Read the problem carefully
2. Solve it step by step
3. Provide ONLY the final numerical answer or result
4. Do NOT include explanations, just the answer

Answer:
EOF

echo "Created solve_problem.txt"

# ============= 2. extract_and_remove.txt =============
cat > "$PROMPT_DIR/extract_and_remove.txt" << 'EOF'
Analyze the following mathematical problem and generate removal variants.

Original Problem: {original_question}
Ground Truth Answer: {ground_truth}

**Task**:

1. Identify ALL conditions in the problem
2. For EACH condition, create a variant where that condition is removed
3. Rewrite the problem without that specific condition

**Instructions**:

- A condition is any piece of information that could help solve the problem
- Each variant should remove exactly ONE condition
- The rewritten problem should be natural and grammatically correct
- Keep all other conditions unchanged in each variant

**Example**:

Original Problem: "Jason bought 1 pencil, Mike bought 2 pencils. How many pencils did Jason buy?"

Conditions identified:
1. Jason bought 1 pencil
2. Mike bought 2 pencils

Variants:

Variant 1 (remove condition 1):
- Removed: "Jason bought 1 pencil"
- Remaining: ["Mike bought 2 pencils"]
- Incomplete Question: "Mike bought 2 pencils. How many pencils did Jason buy?"

Variant 2 (remove condition 2):
- Removed: "Mike bought 2 pencils"  
- Remaining: ["Jason bought 1 pencil"]
- Incomplete Question: "Jason bought 1 pencil. How many pencils did Jason buy?"

**CRITICAL OUTPUT FORMAT**:

Output ONLY a JSON array of variants. Each variant must have:
- removed_condition: the condition that was removed
- remaining_conditions: array of conditions that remain
- incomplete_question: the rewritten problem without the removed condition

Use double curly braces for JSON structure: {{{{ }}}}

Example output:
[
  {{{{
    "removed_condition": "Jason bought 1 pencil",
    "remaining_conditions": ["Mike bought 2 pencils"],
    "incomplete_question": "Mike bought 2 pencils. How many pencils did Jason buy?"
  }}}},
  {{{{
    "removed_condition": "Mike bought 2 pencils",
    "remaining_conditions": ["Jason bought 1 pencil"],
    "incomplete_question": "Jason bought 1 pencil. How many pencils did Jason buy?"
  }}}}
]

Generate all variants:
EOF

echo "Created extract_and_remove.txt"

echo ""
echo "All prompt files created successfully!"
echo "Files created in: $PROMPT_DIR"
ls -la "$PROMPT_DIR"

