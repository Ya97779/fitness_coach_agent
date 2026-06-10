#!/bin/bash
# FitCoach AI 服务器更新脚本
# 用法：在服务器上执行 bash update.sh

set -e

PROJECT_DIR="/var/www/fitcoach/fitness_coach_agent"
VENV_DIR="/var/www/fitcoach/venv"

cd "$PROJECT_DIR"

echo "==> 拉取最新代码..."
# 尝试使用 ghproxy 镜像加速（国内服务器）
GITHUB_MIRROR="https://ghproxy.com/"
ORIGINAL_URL=$(git remote get-url origin)
if [[ "$ORIGINAL_URL" == https://github.com/* ]]; then
    MIRROR_URL="${GITHUB_MIRROR}${ORIGINAL_URL}"
    git remote set-url origin "$MIRROR_URL"
    echo "使用镜像: $MIRROR_URL"
fi
git fetch --depth 1 origin deploy
git reset --hard origin/deploy
# 恢复原始 URL
git remote set-url origin "$ORIGINAL_URL"

echo "==> 安装依赖..."
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt -q

echo "==> 重启后端..."
sudo systemctl restart fitcoach

echo "==> 检查状态..."
sleep 2
sudo systemctl status fitcoach --no-pager -l | head -10

echo "==> 部署完成"
