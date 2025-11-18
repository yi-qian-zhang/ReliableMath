#!/bin/bash
#
# 脚本: 创建矛盾条件生成的prompt目录结构
# 用途: 将所有需要的prompt文件复制到独立目录
#

# 目标目录
TARGET_DIR="/data2/yiqianzhang/ReliableMath/prompt/contradict_data"

# 源目录
SOURCE_DIR="/home/user/ReliableMath/prompt/v4-comp/rewrite"

echo "=========================================="
echo "矛盾条件 Prompt 文件部署脚本"
echo "=========================================="
echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo ""

# 创建目标目录
echo "[1/3] 创建目标目录..."
mkdir -p "$TARGET_DIR"
if [ $? -eq 0 ]; then
    echo "✓ 目录创建成功: $TARGET_DIR"
else
    echo "✗ 目录创建失败！"
    exit 1
fi

echo ""
echo "[2/3] 复制prompt文件..."

# 需要复制的文件列表
files=(
    "extract.txt"
    "contradict_analysis.txt"
    "contradict_rewrite.txt"
    "contradict_verify_s1.txt"
    "contradict_verify_s2.txt"
    "contradict_unsolve_s1.txt"
    "contradict_unsolve_s2.txt"
    "contradict_unsolve_s3.txt"
)

# 计数器
success_count=0
fail_count=0

# 复制文件
for file in "${files[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        cp "$SOURCE_DIR/$file" "$TARGET_DIR/"
        if [ $? -eq 0 ]; then
            echo "  ✓ $file"
            ((success_count++))
        else
            echo "  ✗ $file (复制失败)"
            ((fail_count++))
        fi
    else
        echo "  ⚠ $file (源文件不存在)"
        ((fail_count++))
    fi
done

echo ""
echo "[3/3] 验证文件..."
echo ""
ls -lh "$TARGET_DIR"

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo "成功: $success_count 个文件"
echo "失败: $fail_count 个文件"
echo ""
echo "目标目录: $TARGET_DIR"
echo ""

if [ $fail_count -eq 0 ]; then
    echo "✓ 所有文件复制成功！"
    echo ""
    echo "使用方法："
    echo "python code/contradiction_construction.py \\"
    echo "  --dataset aime \\"
    echo "  --prompt_dir $TARGET_DIR"
else
    echo "⚠ 有 $fail_count 个文件复制失败，请检查！"
    exit 1
fi

echo "=========================================="
