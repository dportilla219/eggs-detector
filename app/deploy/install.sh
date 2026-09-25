#!/usr/bin/env bash
# Instala o actualiza la app en una instancia Ubuntu (probado en EC2 con Ubuntu 24.04).
#
#   git clone https://github.com/dportilla219/eggs-detector.git ~/eggs-detector
#   # copiar el split de test (images/ + labels/) a ~/eggs-data/test  (opcional, para la banda)
#   bash ~/eggs-detector/app/deploy/install.sh
#
# Después hay que abrir el puerto 8000 (TCP) en el grupo de seguridad de la instancia.
set -euo pipefail

REPO="$HOME/eggs-detector"
DATA="$HOME/eggs-data"

sudo apt-get install -y -qq python3-venv >/dev/null
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$REPO/app/requirements.txt"

# Evaluación de la tubería completa sobre el test (la muestra la pestaña de métricas)
if [ -d "$DATA/test/images" ] && [ ! -f "$DATA/eval_test.json" ]; then
  (cd "$REPO/app/server" && "$REPO/.venv/bin/python" evaluate.py --split "$DATA/test" --out "$DATA/eval_test.json")
fi

sudo cp "$REPO/app/deploy/eggs-detector.service" /etc/systemd/system/eggs-detector.service
sudo systemctl daemon-reload
sudo systemctl enable --now eggs-detector.service
sudo systemctl restart eggs-detector.service
sleep 3
curl -fsS http://127.0.0.1:8000/api/health && echo
