import json, re, os

src = "/data/home/zyq/ReliableMath/data/solve/polaris-data-53K.jsonl"
dst = "/data/home/zyq/ReliableMath/data/solve/polaris_easy_20.json"

# 允许形如 "7/8"、" 7/8 "，也兼容纯数字 "7"
def parse_diff_frac(d):
    if isinstance(d, (int, float)):
        return (int(d), None)  # (分子, 分母未知)
    if isinstance(d, str):
        s = d.strip()
        m = re.match(r'^(\d+)\s*/\s*(\d+)$', s)
        if m:
            num, den = map(int, m.groups())
            return (num, den)
        m2 = re.match(r'^\d+(\.\d+)?$', s)
        if m2:
            return (int(float(s)), None)
    return (None, None)

def is_allowed(d):
    num, den = parse_diff_frac(d)
    if num is None:
        return False
    # 精确要求：6/8、7/8、8/8；同时兼容写成纯 6/7/8 的情况
    if den is None:
        return num in (6, 7, 8)
    return den == 8 and num in (6, 7, 8)

picked = []
with open(src, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if is_allowed(obj.get("difficulty")):
            picked.append(obj)
            if len(picked) == 20:
                break

os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, 'w', encoding='utf-8') as f:
    json.dump(picked, f, ensure_ascii=False, indent=2)

print(f"已写出 {len(picked)} 条到 {dst}")
