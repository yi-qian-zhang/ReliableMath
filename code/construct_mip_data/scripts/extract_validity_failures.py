#!/usr/bin/env python3

"""

提取Round A中validity检查失败的案例

"""

import json

import sys

import os

 

def extract_validity_failures(final_json_path, output_path=None):

    """

    从final_n1.json中提取validity_passed=False的案例

 

    Args:

        final_json_path: final_n1.json文件的路径

        output_path: 输出文件路径，默认为validity_analysis.json

    """

    # 检查文件是否存在

    if not os.path.exists(final_json_path):

        print(f"❌ 文件不存在: {final_json_path}")

        sys.exit(1)

 

    # 读取数据

    print(f"📖 正在读取: {final_json_path}")

    with open(final_json_path, 'r', encoding='utf-8') as f:

        dataset = json.load(f)

 

    print(f"✓ 读取了 {len(dataset)} 个原始问题")

 

    # 提取validity失败的案例

    validity_failures = []

    total_variants = 0

 

    for item in dataset:

        original_id = item.get("id", "unknown")

        original_question = item.get("question", "")

 

        for variant in item.get("removal_variants", []):

            total_variants += 1

            variant_id = variant.get("variant_id", "")

            incomplete_question = variant.get("incomplete_question", "")

            removed_conditions = variant.get("removed_conditions", [])

 

            # 获取LLM验证结果

            llm_verification = variant.get("llm_verification", {})

            validity_passed = llm_verification.get("validity_passed", None)

            validity_analysis = llm_verification.get("validity_analysis", "")

            correctness_passed = llm_verification.get("correctness_passed", None)

 

            # 只提取validity_passed=False的案例

            if validity_passed is False:

                failure_case = {

                    "variant_id": variant_id,

                    "original_id": original_id,

                    "original_question": original_question,

                    "removed_conditions": removed_conditions,

                    "incomplete_question": incomplete_question,

                    "validity_passed": validity_passed,

                    "validity_analysis": validity_analysis,

                    "correctness_passed": correctness_passed

                }

                validity_failures.append(failure_case)

 

    print(f"✓ 总共 {total_variants} 个变体")

    print(f"✓ 发现 {len(validity_failures)} 个validity失败案例")

 

    # 确定输出路径

    if output_path is None:

        base_dir = os.path.dirname(final_json_path)

        output_path = os.path.join(base_dir, "validity_analysis.json")

 

    # 保存结果

    with open(output_path, 'w', encoding='utf-8') as f:

        json.dump(validity_failures, f, ensure_ascii=False, indent=2)

 

    print(f"✓ 已保存到: {output_path}")

 

    # 统计分析

    print("\n" + "="*70)

    print("📊 Validity失败原因统计")

    print("="*70)

 

    # 简单统计（基于关键词）

    issue_keywords = {

        "Issue 1 (Question Stem Deleted)": ["question stem", "no longer asking"],

        "Issue 2 (Dangling References)": ["dangling", "pronoun", "reference", "this condition", "antecedent"],

        "Issue 3 (Missing Context)": ["context", "scenario", "background"],

        "Issue 4 (Infinite Solutions)": ["infinite", "infinitely many", "too many"],

        "Issue 5 (Trivially Solvable)": ["trivial", "still solvable", "can be solved"]

    }

 

    issue_counts = {issue: 0 for issue in issue_keywords.keys()}

 

    for case in validity_failures:

        analysis_lower = case["validity_analysis"].lower()

        for issue_name, keywords in issue_keywords.items():

            if any(keyword.lower() in analysis_lower for keyword in keywords):

                issue_counts[issue_name] += 1

                break  # 每个案例只计入第一个匹配的issue

 

    for issue_name, count in issue_counts.items():

        if count > 0:

            percentage = count / len(validity_failures) * 100

            print(f"{issue_name}: {count} ({percentage:.1f}%)")

 

    print("\n前3个失败案例示例:")

    print("="*70)

    for i, case in enumerate(validity_failures[:3], 1):

        print(f"\n【案例 {i}】 {case['variant_id']}")

        print(f"原始问题: {case['original_question'][:100]}...")

        print(f"移除条件: {case['removed_conditions']}")

        print(f"改写后: {case['incomplete_question'][:100]}...")

        print(f"失败原因: {case['validity_analysis'][:200]}...")

 

    return validity_failures

 

 

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("用法: python extract_validity_failures.py <final_n1.json路径> [输出路径]")

        print("\n示例:")

        print("  python extract_validity_failures.py data/xxx/polaris_normal_10times7_final_n1.json")

        print("  python extract_validity_failures.py data/xxx/polaris_normal_10times7_final_n1.json my_analysis.json")

        sys.exit(1)

 

    final_path = sys.argv[1]

    output_path = sys.argv[2] if len(sys.argv) > 2 else None

 

    extract_validity_failures(final_path, output_path)