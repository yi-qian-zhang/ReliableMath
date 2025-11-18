# Prompt文件生成脚本使用说明

## 📋 概述

`deploy_contradiction_prompts.sh` 是一个自动化脚本，用于在生产环境直接生成所有矛盾条件prompt文件，无需从开发环境复制。

## 🎯 目标位置

```
/data2/yiqianzhang/ReliableMath/prompt/contradict_data/
├── extract.txt                      # Step 1: 条件提取
├── contradict_analysis.txt          # Step 2.1: 分析矛盾
├── contradict_rewrite.txt           # Step 2.2: 重写问题
├── contradict_verify_s1.txt         # Step 3.1: 验证单条件
├── contradict_verify_s2.txt         # Step 3.2: 提取矛盾描述
├── contradict_unsolve_s1.txt        # Step 3.3: 分析不可解性
├── contradict_unsolve_s2.txt        # Step 3.4: 判断不可解
└── contradict_unsolve_s3.txt        # Step 3.5: 提取原因
```

**共8个文件**

## 🚀 使用方法

### 方法1: 直接运行（推荐）

```bash
cd /home/user/ReliableMath

# 运行脚本
./deploy_contradiction_prompts.sh
```

### 方法2: 从任意位置运行

```bash
bash /home/user/ReliableMath/deploy_contradiction_prompts.sh
```

### 方法3: 复制到生产环境运行

```bash
# 复制脚本到生产环境
cp /home/user/ReliableMath/deploy_contradiction_prompts.sh \
   /data2/yiqianzhang/ReliableMath/

# 在生产环境运行
cd /data2/yiqianzhang/ReliableMath
./deploy_contradiction_prompts.sh
```

## 📊 脚本输出示例

```bash
==========================================
矛盾条件 Prompt 文件生成脚本
==========================================
目标目录: /data2/yiqianzhang/ReliableMath/prompt/contradict_data

[1/9] 创建目标目录...
✓ 目录创建成功

[2/9] 生成 extract.txt...
✓ extract.txt 创建成功

[3/9] 生成 contradict_analysis.txt...
✓ contradict_analysis.txt 创建成功

[4/9] 生成 contradict_rewrite.txt...
✓ contradict_rewrite.txt 创建成功

[5/9] 生成 contradict_verify_s1.txt...
✓ contradict_verify_s1.txt 创建成功

[6/9] 生成 contradict_verify_s2.txt...
✓ contradict_verify_s2.txt 创建成功

[7/9] 生成 contradict_unsolve_s1.txt...
✓ contradict_unsolve_s1.txt 创建成功

[8/9] 生成 contradict_unsolve_s2.txt...
✓ contradict_unsolve_s2.txt 创建成功

[9/9] 生成 contradict_unsolve_s3.txt...
✓ contradict_unsolve_s3.txt 创建成功

==========================================
生成完成！
==========================================
目标目录: /data2/yiqianzhang/ReliableMath/prompt/contradict_data

生成的文件列表：
total 32K
-rw-r--r-- 1 user user  524 Nov 18 extract.txt
-rw-r--r-- 1 user user  565 Nov 18 contradict_analysis.txt
-rw-r--r-- 1 user user  486 Nov 18 contradict_rewrite.txt
-rw-r--r-- 1 user user  830 Nov 18 contradict_verify_s1.txt
-rw-r--r-- 1 user user  896 Nov 18 contradict_verify_s2.txt
-rw-r--r-- 1 user user  879 Nov 18 contradict_unsolve_s1.txt
-rw-r--r-- 1 user user  665 Nov 18 contradict_unsolve_s2.txt
-rw-r--r-- 1 user user  592 Nov 18 contradict_unsolve_s3.txt

文件数量：
  生成: 8 个文件
  预期: 8 个文件

✓ 所有文件生成成功！

使用方法：
cd /data2/yiqianzhang/ReliableMath

python code/contradiction_construction/contradiction_construction.py \
  --dataset aime \
  --prompt_dir /data2/yiqianzhang/ReliableMath/prompt/contradict_data \
  --test_mode
==========================================
```

## ✅ 验证部署

### 1. 检查文件是否存在

```bash
ls -lh /data2/yiqianzhang/ReliableMath/prompt/contradict_data/
```

### 2. 查看某个文件内容

```bash
cat /data2/yiqianzhang/ReliableMath/prompt/contradict_data/extract.txt
```

### 3. 统计文件数量

```bash
ls -1 /data2/yiqianzhang/ReliableMath/prompt/contradict_data/ | wc -l
# 应该输出: 8
```

## 🔄 重新生成

如果需要重新生成（例如prompt内容有更新）：

```bash
# 删除旧文件
rm -rf /data2/yiqianzhang/ReliableMath/prompt/contradict_data

# 重新运行脚本
./deploy_contradiction_prompts.sh
```

## 📝 Prompt文件说明

### Step 1: extract.txt
- **用途**: 从原始问题中提取所有关键条件
- **输入**: `{original_math_question}`
- **输出**: JSON数组 `["条件1", "条件2", ...]`
- **模型**: gpt-4o-mini

### Step 2.1: contradict_analysis.txt
- **用途**: 分析如何为某个条件添加矛盾
- **输入**: `{original_math_question}`, `{original_answer}`, `{extracted_condition}`
- **输出**: 分析文本
- **模型**: DeepSeek-R1-Distill-Qwen-7B

### Step 2.2: contradict_rewrite.txt
- **用途**: 生成添加矛盾后的问题
- **输入**: `{original_math_question}`, `{original_answer}`, `{extracted_condition}`
- **输出**: 重写后的问题
- **模型**: DeepSeek-R1-Distill-Qwen-7B

### Step 3.1: contradict_verify_s1.txt
- **用途**: 验证是否只修改了一个条件
- **输入**: `{original_question}`, `{rewritten_question}`
- **输出**: True/False
- **模型**: gpt-4o-mini

### Step 3.2: contradict_verify_s2.txt
- **用途**: 提取矛盾条件的描述
- **输入**: `{original_question}`, `{original_condition}`, `{rewritten_question}`
- **输出**: 矛盾描述文本
- **模型**: DeepSeek-R1-Distill-Qwen-7B

### Step 3.3: contradict_unsolve_s1.txt
- **用途**: 分析为什么问题不可解
- **输入**: `{original_question}`, `{original_answer}`, `{rewritten_question}`
- **输出**: 分析文本（2-5句话）
- **模型**: DeepSeek-R1-Distill-Qwen-7B

### Step 3.4: contradict_unsolve_s2.txt
- **用途**: 判断是否真的不可解
- **输入**: `{original_question}`, `{original_answer}`, `{rewritten_question}`, `{unsolvability_analysis}`
- **输出**: True/False
- **模型**: DeepSeek-R1-Distill-Qwen-7B

### Step 3.5: contradict_unsolve_s3.txt
- **用途**: 提取简洁的不可解原因
- **输入**: `{original_question}`, `{rewritten_question}`, `{unsolvability_analysis}`
- **输出**: 简洁原因（1-2句话）
- **模型**: deepseek-v3

## 🔧 自定义修改

### 修改目标目录

编辑脚本第7行：

```bash
# 修改前
TARGET_DIR="/data2/yiqianzhang/ReliableMath/prompt/contradict_data"

# 修改后
TARGET_DIR="/your/custom/path/prompt/contradict_data"
```

### 修改Prompt内容

编辑脚本中对应的 `cat > "$TARGET_DIR/xxx.txt" << 'PROMPT_EOF'` 部分。

例如修改 `extract.txt`:

```bash
cat > "$TARGET_DIR/extract.txt" << 'PROMPT_EOF'
你的新prompt内容...
{original_math_question}
...
PROMPT_EOF
```

## ⚠️ 注意事项

1. **目标目录会被创建**: 如果目录不存在，脚本会自动创建
2. **文件会被覆盖**: 如果文件已存在，会被新内容覆盖
3. **权限检查**: 确保对目标目录有写权限
4. **编码格式**: 所有文件使用UTF-8编码

## 🆚 与复制脚本的对比

| 特性 | deploy_contradiction_prompts.sh | setup_contradiction_prompts.sh |
|-----|--------------------------------|-------------------------------|
| 方式 | 直接生成文件 | 从源目录复制 |
| 依赖 | 无需源文件 | 需要开发环境 |
| 速度 | 快 | 较快 |
| 灵活性 | 高（可自定义） | 低（依赖源） |
| 适用场景 | 生产部署 | 开发→生产 |

## 📚 相关文档

- [使用指南](CONTRADICTION_USAGE.md) - 完整的使用文档
- [Prompt说明](PROMPT_README.md) - Prompt文件详细说明
- [部署指南](DEPLOYMENT.md) - 部署步骤
- [复制脚本](setup_contradiction_prompts.sh) - 从开发环境复制

---

**最后更新**: 2025-11-18
**维护者**: Claude Code
