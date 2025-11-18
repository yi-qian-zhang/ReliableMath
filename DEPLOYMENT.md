# ReliableMath 矛盾条件生成 - 部署指南

## 📦 部署位置

### 方案A: 开发环境（当前）
```
/home/user/ReliableMath/
├── code/
│   ├── contradiction_construction.py  ← 主程序
│   ├── deepscaler/                     ← 依赖模块
│   │   ├── rewards/
│   │   │   └── math_utils/
│   │   │       └── utils.py
│   │   └── system_prompts.py
│   └── ...
├── prompt/v4-comp/rewrite/             ← Prompt文件
└── data/
```

### 方案B: 生产环境
```
/data2/yiqianzhang/ReliableMath/
├── code/
│   ├── contradiction_construction/
│   │   └── contradiction_construction.py  ← 主程序
│   ├── deepscaler/                         ← 依赖模块
│   │   ├── rewards/
│   │   │   └── math_utils/
│   │   │       └── utils.py
│   │   └── system_prompts.py
│   └── ...
├── prompt/contradict_data/                 ← Prompt文件
└── data/
```

---

## 🔧 部署步骤

### 1. 复制代码到生产环境

```bash
# 创建目录结构
mkdir -p /data2/yiqianzhang/ReliableMath/code/contradiction_construction

# 复制主程序
cp /home/user/ReliableMath/code/contradiction_construction.py \
   /data2/yiqianzhang/ReliableMath/code/contradiction_construction/

# 复制deepscaler模块（如果还没有的话）
cp -r /home/user/ReliableMath/code/deepscaler \
      /data2/yiqianzhang/ReliableMath/code/

# 复制其他必要的代码文件
cp -r /home/user/ReliableMath/code/metrics \
      /data2/yiqianzhang/ReliableMath/code/
```

### 2. 部署Prompt文件

使用提供的脚本：

```bash
cd /home/user/ReliableMath

# 运行部署脚本
./setup_contradiction_prompts.sh
```

或手动复制：

```bash
mkdir -p /data2/yiqianzhang/ReliableMath/prompt/contradict_data

cp /home/user/ReliableMath/prompt/v4-comp/rewrite/extract.txt \
   /home/user/ReliableMath/prompt/v4-comp/rewrite/contradict_*.txt \
   /data2/yiqianzhang/ReliableMath/prompt/contradict_data/
```

### 3. 安装依赖

```bash
# 进入生产环境
cd /data2/yiqianzhang/ReliableMath

# 安装Python依赖
pip install openai tqdm tiktoken pylatexenc sympy

# 如果使用conda环境
conda install -c conda-forge pylatexenc sympy
```

### 4. 配置API密钥

```bash
# 编辑API配置文件
vim /data2/yiqianzhang/ReliableMath/data/api_keys.json
```

内容示例：
```json
{
  "gpt-4o-mini": [["gpt-4o-mini", "sk-xxx", "https://api.openai.com/v1"]],
  "DeepSeek-R1-Distill-Qwen-7B": [["DeepSeek-R1-Distill-Qwen-7B", "", "http://localhost:8000/v1"]],
  "deepseek-v3": [["deepseek-chat", "sk-xxx", "https://api.deepseek.com"]]
}
```

---

## 🚀 运行方法

### 方案A: 在开发环境运行

```bash
cd /home/user/ReliableMath

python code/contradiction_construction.py \
  --dataset aime \
  --model gpt-4o-mini \
  --analysis_model DeepSeek-R1-Distill-Qwen-7B \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --extract_model deepseek-v3 \
  --data_dir data/solve \
  --output_dir data/construct_contradiction \
  --prompt_dir prompt/v4-comp/rewrite \
  --test_mode
```

### 方案B: 在生产环境运行

```bash
cd /data2/yiqianzhang/ReliableMath

python code/contradiction_construction/contradiction_construction.py \
  --dataset aime \
  --model gpt-4o-mini \
  --analysis_model DeepSeek-R1-Distill-Qwen-7B \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --extract_model deepseek-v3 \
  --data_dir data/solve \
  --output_dir data/construct_contradiction \
  --prompt_dir prompt/contradict_data \
  --temperature 1.0 \
  --max_attempts 8 \
  --threads 8
```

---

## 🔍 路径自动检测

代码会自动检测运行环境并设置正确的import路径：

**检测逻辑：**
```python
# 如果脚本在 contradiction_construction/ 目录下
#   → 添加 /data2/yiqianzhang/ReliableMath/code 到 sys.path
#   → import deepscaler 会查找 /data2/.../code/deepscaler

# 如果脚本在 code/ 目录下
#   → 添加 /home/user/ReliableMath 到 sys.path
#   → 添加 /home/user/ReliableMath/code 到 sys.path
#   → import deepscaler 会查找 /home/user/.../code/deepscaler
```

**验证import是否成功：**
```bash
python -c "
import sys
import os

# 设置工作目录
os.chdir('/data2/yiqianzhang/ReliableMath')

# 运行导入测试
sys.path.insert(0, 'code')
from deepscaler.system_prompts import ORM_PROMPT
print('✓ Import successful!')
"
```

---

## ⚠️ 常见问题

### Q1: ImportError: No module named 'deepscaler'

**原因：** sys.path设置不正确或deepscaler不在预期位置

**解决方案：**
```bash
# 检查deepscaler位置
ls -la /data2/yiqianzhang/ReliableMath/code/deepscaler

# 如果不存在，复制过去
cp -r /home/user/ReliableMath/code/deepscaler \
      /data2/yiqianzhang/ReliableMath/code/
```

### Q2: ModuleNotFoundError: No module named 'pylatexenc'

**原因：** 缺少依赖

**解决方案：**
```bash
pip install pylatexenc sympy
```

### Q3: 找不到prompt文件

**错误信息：**
```
Prompt file not found: /data2/yiqianzhang/ReliableMath/prompt/contradict_data/extract.txt
```

**解决方案：**
```bash
# 运行部署脚本
cd /home/user/ReliableMath
./setup_contradiction_prompts.sh

# 或指定正确的prompt目录
python code/contradiction_construction/contradiction_construction.py \
  --prompt_dir /data2/yiqianzhang/ReliableMath/prompt/contradict_data
```

### Q4: API keys not found

**错误信息：**
```
api_keys.json not found at data/api_keys.json!
```

**解决方案：**
```bash
# 确保在正确的目录运行
cd /data2/yiqianzhang/ReliableMath

# 检查api_keys.json是否存在
ls -la data/api_keys.json

# 如果不存在，创建或复制
cp /home/user/ReliableMath/data/api_keys.json \
   /data2/yiqianzhang/ReliableMath/data/
```

---

## 📋 检查清单

部署前请确认：

- [ ] deepscaler模块已复制到正确位置
- [ ] 所有prompt文件已部署到 `prompt/contradict_data/`
- [ ] Python依赖已安装（openai, tqdm, tiktoken, pylatexenc, sympy）
- [ ] API密钥配置文件已创建并配置正确
- [ ] 输入数据已准备在 `data/solve/`
- [ ] 输出目录已创建：`data/construct_contradiction/`
- [ ] 本地vLLM服务已启动（如使用本地模型）

---

## 🧪 验证部署

```bash
cd /data2/yiqianzhang/ReliableMath

# 测试导入
python -c "
import sys
sys.path.insert(0, 'code')
from deepscaler.system_prompts import ORM_PROMPT
print('✓ Import OK')
"

# 测试运行（只处理1个样本）
python code/contradiction_construction/contradiction_construction.py \
  --dataset aime \
  --test_mode \
  --threads 1

# 检查输出
ls -lh data/construct_contradiction/
```

---

## 📚 相关文档

- [使用指南](CONTRADICTION_USAGE.md) - 完整的使用文档
- [Prompt说明](PROMPT_README.md) - Prompt文件详细说明
- [部署脚本](setup_contradiction_prompts.sh) - 自动化部署工具

---

**最后更新**: 2025-11-18
**维护者**: Claude Code
