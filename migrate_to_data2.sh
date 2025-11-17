#!/bin/bash

 

echo "========================================="

echo "主目录迁移到 /data2/yiqianzhang"

echo "========================================="

echo ""

 

# 颜色定义

RED='\033[0;31m'

GREEN='\033[0;32m'

YELLOW='\033[1;33m'

NC='\033[0m' # No Color

 

# 检查目标目录

TARGET_DIR="/data2/yiqianzhang"

HOME_DIR="$HOME"

 

echo "1. 检查磁盘空间..."

echo ""

echo "当前磁盘使用情况:"

df -h | grep -E 'nvme0n1|data2|Filesystem'

echo ""

 

echo "主目录大小分析:"

du -sh "$HOME_DIR"

echo ""

 

echo "主目录下各文件夹大小:"

du -sh "$HOME_DIR"/* 2>/dev/null | sort -hr

echo ""

 

# 检查目标目录

if [ ! -d "$TARGET_DIR" ]; then

    echo -e "${YELLOW}目标目录 $TARGET_DIR 不存在${NC}"

    read -p "是否创建目标目录? (y/n): " create_dir

    if [ "$create_dir" = "y" ]; then

        mkdir -p "$TARGET_DIR"

        echo -e "${GREEN}已创建目录: $TARGET_DIR${NC}"

    else

        echo -e "${RED}取消操作${NC}"

        exit 1

    fi

fi

 

echo ""

echo "========================================="

echo "建议迁移的目录（大型项目和数据）"

echo "========================================="

echo ""

echo "推荐迁移的目录:"

echo "1) AR-Bench/"

echo "2) Interactive-R1/"

echo "3) ReliableMath/"

echo "4) UserSim/"

echo "5) .vscode-server/"

echo "6) .cache/"

echo "7) .triton/"

echo "8) .npm/"

echo ""

echo -e "${YELLOW}不建议迁移的目录:${NC}"

echo "- .ssh/ (SSH配置)"

echo "- .bashrc, .profile (Shell配置)"

echo "- .gitconfig, .git-credentials (Git配置)"

echo "- .conda/ (Conda配置)"

echo ""

 

echo "========================================="

echo "迁移选项"

echo "========================================="

echo "1) 自动迁移大型项目目录（AR-Bench, Interactive-R1, ReliableMath, UserSim）"

echo "2) 迁移VSCode相关（.vscode-server）"

echo "3) 迁移缓存目录（.cache, .triton, .npm）"

echo "4) 自定义选择迁移"

echo "5) 仅显示迁移建议，不执行"

echo "6) 退出"

echo ""

read -p "请选择操作 (1-6): " choice

 

# 迁移函数

migrate_dir() {

    local dir_name="$1"

    local source_path="$HOME_DIR/$dir_name"

    local target_path="$TARGET_DIR/$dir_name"

 

    if [ ! -e "$source_path" ]; then

        echo -e "${YELLOW}跳过: $dir_name (不存在)${NC}"

        return

    fi

 

    if [ -L "$source_path" ]; then

        echo -e "${YELLOW}跳过: $dir_name (已是软链接)${NC}"

        return

    fi

 

    echo -e "${GREEN}迁移: $dir_name${NC}"

    echo "  源路径: $source_path"

    echo "  目标路径: $target_path"

 

    # 计算大小

    size=$(du -sh "$source_path" 2>/dev/null | cut -f1)

    echo "  大小: $size"

 

    read -p "  确认迁移? (y/n): " confirm

    if [ "$confirm" != "y" ]; then

        echo "  跳过"

        return

    fi

 

    # 复制文件

    echo "  正在复制..."

    rsync -av --progress "$source_path" "$TARGET_DIR/" 2>&1 | tail -5

 

    if [ $? -eq 0 ]; then

        # 验证复制

        echo "  验证复制完整性..."

        source_count=$(find "$source_path" -type f 2>/dev/null | wc -l)

        target_count=$(find "$target_path" -type f 2>/dev/null | wc -l)

 

        if [ "$source_count" -eq "$target_count" ]; then

            echo -e "  ${GREEN}验证成功 (文件数: $source_count)${NC}"

 

            # 备份原目录

            backup_path="${source_path}.backup.$(date +%Y%m%d_%H%M%S)"

            echo "  创建备份: $backup_path"

            mv "$source_path" "$backup_path"

 

            # 创建软链接

            echo "  创建软链接..."

            ln -s "$target_path" "$source_path"

 

            if [ -L "$source_path" ]; then

                echo -e "  ${GREEN}✓ 迁移成功！${NC}"

                echo "  备份保存在: $backup_path"

                echo "  提示: 确认无误后可删除备份: rm -rf $backup_path"

            else

                echo -e "  ${RED}✗ 软链接创建失败${NC}"

                mv "$backup_path" "$source_path"

            fi

        else

            echo -e "  ${RED}✗ 验证失败 (源:$source_count, 目标:$target_count)${NC}"

            echo "  保留原文件，请手动检查"

        fi

    else

        echo -e "  ${RED}✗ 复制失败${NC}"

    fi

    echo ""

}

 

case $choice in

    1)

        echo "开始迁移大型项目目录..."

        for dir in "AR-Bench" "Interactive-R1" "ReliableMath" "UserSim"; do

            migrate_dir "$dir"

        done

        ;;

    2)

        echo "开始迁移VSCode相关目录..."

        migrate_dir ".vscode-server"

        ;;

    3)

        echo "开始迁移缓存目录..."

        for dir in ".cache" ".triton" ".npm"; do

            migrate_dir "$dir"

        done

        ;;

    4)

        echo "请输入要迁移的目录名（相对于主目录），多个目录用空格分隔:"

        read -p "> " custom_dirs

        for dir in $custom_dirs; do

            migrate_dir "$dir"

        done

        ;;

    5)

        echo ""

        echo "========================================="

        echo "迁移建议"

        echo "========================================="

        echo ""

        echo "手动迁移命令示例:"

        echo ""

        echo "# 1. 创建目标目录"

        echo "mkdir -p /data2/yiqianzhang"

        echo ""

        echo "# 2. 复制目录（以ReliableMath为例）"

        echo "rsync -av ~/ReliableMath /data2/yiqianzhang/"

        echo ""

        echo "# 3. 验证复制成功后，备份原目录"

        echo "mv ~/ReliableMath ~/ReliableMath.backup"

        echo ""

        echo "# 4. 创建软链接"

        echo "ln -s /data2/yiqianzhang/ReliableMath ~/ReliableMath"

        echo ""

        echo "# 5. 验证软链接"

        echo "ls -l ~/ReliableMath"

        echo ""

        ;;

    *)

        echo "退出"

        exit 0

        ;;

esac

 

echo ""

echo "========================================="

echo "迁移后检查"

echo "========================================="

echo ""

echo "磁盘使用情况:"

df -h | grep -E 'nvme0n1|data2|Filesystem'

echo ""

echo "软链接检查:"

ls -l "$HOME_DIR" | grep -E '^l'

echo ""

echo "完成！"
