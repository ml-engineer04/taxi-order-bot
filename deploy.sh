#!/usr/bin/env bash
# ==============================================================================
# Yo'lchi AI - Production Safe Deployment Script
# ==============================================================================
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ==============================================================================

set -euo pipefail

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE} 🚀 Yo'lchi AI: Production Deployment Started        ${NC}"
echo -e "${BLUE}======================================================${NC}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Check Virtual Environment
echo -e "\n${YELLOW}[1/6] Virtual muhit tekshirilmoqda...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  .venv topilmadi, yangi virtual muhit yaratilmoqda...${NC}"
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Pull Latest Code from GitHub
echo -e "\n${YELLOW}[2/6] GitHub'dan so'nggi o'zgarishlar tortib olinmoqda...${NC}"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    CURRENT_BRANCH=$(git branch --show-current || echo "main")
    echo -e "Tarmoq: ${GREEN}${CURRENT_BRANCH}${NC}"
    git pull origin "$CURRENT_BRANCH"
else
    echo -e "${YELLOW}⚠️  Git repozitoriy emas, pull o'tkazib yuborildi.${NC}"
fi

# 3. Update Dependencies
echo -e "\n${YELLOW}[3/6] Python paketlari yangilanmoqda (requirements.txt)...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt --quiet
fi

# 4. Run Automated Test Suite
echo -e "\n${YELLOW}[4/6] Avtomatik testlar ishga tushirilmoqda (pytest)...${NC}"
if pytest tests/ -q; then
    echo -e "${GREEN}✅ Barcha testlar muvaffaqiyatli o'tdi!${NC}"
else
    echo -e "${RED}❌ XATOLIK: Testlar muvaffaqiyatsiz yakunlandi!${NC}"
    echo -e "${RED}Bot xavfsizlik maqsadida RESTART QILINMADI.${NC}"
    echo -e "${YELLOW}Iltimos, koddagi xatoni to'g'irlab qayta urinib ko'ring.${NC}"
    exit 1
fi

# 5. Database Schema & WAL Verification
echo -e "\n${YELLOW}[5/6] Ma'lumotlar bazasi tekshirilmoqda...${NC}"
python3 -c "
import asyncio
from app.database.session import init_db, close_db
async def check():
    await init_db()
    await close_db()
asyncio.run(check())
print('Baza sozlamalari tayyor.')
"

# 6. Graceful Service Restart (Systemd)
echo -e "\n${YELLOW}[6/6] Bot xizmati qayta ishga tushirilmoqda (Systemd)...${NC}"
SERVICE_NAME="taxi_bot"

if command -v systemctl >/dev/null 2>&1 && (systemctl cat "${SERVICE_NAME}.service" >/dev/null 2>&1 || [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]); then
    echo -e "Servis qayta ishga tushirilmoqda: ${SERVICE_NAME}..."
    sudo systemctl restart "${SERVICE_NAME}"
    sleep 2

    if sudo systemctl is-active --quiet "${SERVICE_NAME}"; then
        echo -e "${GREEN}======================================================${NC}"
        echo -e "${GREEN} 🎉 BOT MUVAFFAQIYATLI ISHGA TUSHDI VA ISHLAMOQDA!   ${NC}"
        echo -e "${GREEN}======================================================${NC}"
        echo -e "\n${BLUE}So'nggi loglar:${NC}"
        sudo journalctl -u "${SERVICE_NAME}" -n 15 --no-pager
    else
        echo -e "${RED}❌ Xatolik: Servis ishga tushmadi! Loglarni tekshiring:${NC}"
        sudo journalctl -u "${SERVICE_NAME}" -n 25 --no-pager
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  Systemd xizmati (${SERVICE_NAME}.service) topilmadi.${NC}"
    echo -e "${YELLOW}Botni qo'lda ishga tushirish uchun: source .venv/bin/activate && python -m app.main${NC}"
fi
