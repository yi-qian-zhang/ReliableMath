#!/bin/bash

echo "========================================="
echo "VSCode Remote 连接问题诊断和修复工具"
echo "========================================="
echo ""

# 检查是否有残留的vscode-server进程
echo "1. 检查vscode-server进程..."
VSCODE_PROCESSES=$(ps aux | grep vscode-server | grep -v grep)
if [ -n "$VSCODE_PROCESSES" ]; then
    echo "发现以下vscode-server进程："
    echo "$VSCODE_PROCESSES"
    echo ""
    read -p "是否要杀死这些进程？(y/n): " kill_processes
    if [ "$kill_processes" = "y" ]; then
        pkill -f vscode-server
        echo "已杀死vscode-server进程"
    fi
else
    echo "没有发现运行中的vscode-server进程"
fi
echo ""

# 检查.vscode-server目录
echo "2. 检查.vscode-server目录..."
if [ -d ~/.vscode-server ]; then
    echo "目录大小："
    du -sh ~/.vscode-server
    echo ""
    echo "目录内容："
    ls -lh ~/.vscode-server
    echo ""

    # 检查日志文件
    echo "3. 检查最新的日志文件..."
    LOG_FILES=$(find ~/.vscode-server -name "*.log" -type f)
    if [ -n "$LOG_FILES" ]; then
        for log in $LOG_FILES; do
            echo "--- $log ---"
            size=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log" 2>/dev/null)
            if [ "$size" -gt 0 ]; then
                tail -20 "$log"
            else
                echo "(空文件)"
            fi
            echo ""
        done
    fi

    # 检查权限
    echo "4. 检查目录权限..."
    ls -ld ~/.vscode-server
    echo ""

    # 提供清理选项
    echo "========================================="
    echo "修复选项："
    echo "========================================="
    echo "1) 清理旧的日志和锁文件（推荐首先尝试）"
    echo "2) 完全删除.vscode-server目录并重新安装"
    echo "3) 仅备份并退出"
    echo "4) 退出不做任何更改"
    echo ""
    read -p "请选择操作 (1-4): " choice

    case $choice in
        1)
            echo "清理旧的日志和锁文件..."
            rm -f ~/.vscode-server/.*.log
            rm -f ~/.vscode-server/.*.pid
            rm -f ~/.vscode-server/.*.token
            echo "清理完成！请尝试重新连接VSCode。"
            ;;
        2)
            echo "备份当前.vscode-server目录..."
            BACKUP_DIR=~/.vscode-server.backup.$(date +%Y%m%d_%H%M%S)
            mv ~/.vscode-server "$BACKUP_DIR"
            echo "已备份到: $BACKUP_DIR"
            echo "已删除.vscode-server目录。"
            echo "请尝试重新连接VSCode，它会自动重新安装。"
            ;;
        3)
            echo "创建备份..."
            BACKUP_DIR=~/.vscode-server.backup.$(date +%Y%m%d_%H%M%S)
            cp -r ~/.vscode-server "$BACKUP_DIR"
            echo "已备份到: $BACKUP_DIR"
            ;;
        *)
            echo "未做任何更改。"
            ;;
    esac
else
    echo ".vscode-server目录不存在！"
    echo "请尝试用VSCode连接，它会自动创建。"
fi

echo ""
echo "========================================="
echo "额外建议："
echo "========================================="
echo "1. 确保本地VSCode版本是最新的"
echo "2. 检查SSH配置文件 ~/.ssh/config"
echo "3. 尝试在VSCode中使用 'Remote-SSH: Kill VS Code Server on Host' 命令"
echo "4. 检查远程服务器磁盘空间: df -h"
echo ""
echo "完成！"
