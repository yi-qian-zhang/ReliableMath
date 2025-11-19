# 简单分析脚本
import json

# 读取最终数据
data = json.load(open("/data2/yiqianzhang/ReliableMath/data/DeepSeek-R1-Distill-Qwen-32B-8715/11-19/official_mode/missing_one/polaris_normal_10times7/polaris_normal_10times7_final_n1.json"))

# 统计Round A失败原因
round_a_failures = {
    "correctness_fail": [],
    "validity_fail": []
}

for item in data:
    for variant in item.get("removal_variants", []):
        llm_verif = variant.get("llm_verification", {})
        if not llm_verif.get("correctness_passed", True):
            round_a_failures["correctness_fail"].append({
                "id": variant["variant_id"],
                "analysis": llm_verif.get("correctness_analysis", "")
            })
        if not llm_verif.get("validity_passed", True):
            round_a_failures["validity_fail"].append({
                "id": variant["variant_id"],
                "analysis": llm_verif.get("validity_analysis", ""),
                "incomplete_question": variant["incomplete_question"]
            })

# 查看几个validity失败的例子
print(f"Validity failures: {len(round_a_failures['validity_fail'])}")
for case in round_a_failures["validity_fail"][:3]:
    print(f"\n{case['id']}:")
    print(f"Question: {case['incomplete_question']}")
    print(f"Reason: {case['analysis'][:200]}...")