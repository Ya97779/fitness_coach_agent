#!/bin/bash
# FitCoach AI 服务器更新脚本
# 用法：在服务器上执行 bash update.sh

set -e

PROJECT_DIR="/var/www/fitcoach/fitness_coach_agent"
VENV_DIR="/var/www/fitcoach/venv"

cd "$PROJECT_DIR"

echo "==> 拉取最新代码..."
git fetch --depth 1 origin deploy
git reset --hard origin/deploy

echo "==> 安装依赖..."
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt -q

echo "==> 检查数据库迁移..."
service_stopped=0
restart_if_stopped() {
    if [ "$service_stopped" -eq 1 ]; then
        echo "==> 迁移未完成，尝试恢复后端服务..."
        sudo systemctl restart fitcoach || true
    fi
}
trap restart_if_stopped EXIT

if "$VENV_DIR/bin/python" scripts/migrate_phase02.py --check; then
    echo "==> 数据库已是最新，跳过迁移"
else
    migration_status=$?
    if [ "$migration_status" -ne 10 ]; then
        echo "==> 数据库迁移检查失败（退出码: $migration_status）"
        exit "$migration_status"
    fi

    echo "==> 检测到待执行迁移，停止后端并迁移..."
    sudo systemctl stop fitcoach
    service_stopped=1
    "$VENV_DIR/bin/python" scripts/migrate_phase02.py
fi

echo "==> 重启后端..."
sudo systemctl restart fitcoach
service_stopped=0
trap - EXIT

echo "==> 检查状态..."
sleep 2
sudo systemctl status fitcoach --no-pager -l | head -10

echo "==> 部署完成"
