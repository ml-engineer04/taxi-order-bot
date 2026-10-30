#!/usr/bin/env bash
# ==============================================================================
# Yo'lchi AI - Server Initial Setup Script (Ubuntu/Debian)
# ==============================================================================
# Usage on fresh VPS:
#   chmod +x setup_server.sh
#   ./setup_server.sh
# ==============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE} 🛠️  Yo'lchi AI: Server Initial Setup               ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. System packages
echo -e "\n${YELLOW}[1/4] Tizim paketlari o'rnatilmoqda (Python, Git, ffmpeg, sqlite3)...${NC}"
sudo apt update -y
sudo apt install -y python3 python3-venv python3-pip git curl htop sqlite3 ffmpeg

# 2. Virtual Environment
echo -e "\n${YELLOW}[2/4] Python virtual muhiti yaratilmoqda (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
fi

# 3. Create required directories
echo -e "\n${YELLOW}[3/4] Kerakli papkalar tekshirilmoqda (data, logs)...${NC}"
mkdir -p data logs

# 4. Configure Systemd Service
echo -e "\n${YELLOW}[4/4] Systemd xizmati sozlanmoqda (taxi_bot.service)...${NC}"
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

# Generate tailored service file
cat <<EOF | sudo tee /etc/systemd/system/taxi_bot.service > /dev/null
[Unit]
Description=Yo'lchi AI Taxi Forwarder Bot & Userbot
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${CURRENT_DIR}
EnvironmentFile=${CURRENT_DIR}/.env
ExecStart=${CURRENT_DIR}/.venv/bin/python -m app.main
Restart=always
RestartSec=5s

# Process management
KillSignal=SIGINT
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable taxi_bot

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} ✅ SERVER MUVAFFAQIYATLI SOZLANDI!                  ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Endi quyidagilarni bajaring:"
echo -e "1. ${YELLOW}.env${NC} faylini tekshiring va to'ldiring."
echo -e "2. ${YELLOW}data/userbot.session${NC} faylini serverga yuklang."
echo -e "3. Botni ishga tushirish uchun: ${GREEN}./deploy.sh${NC}"
