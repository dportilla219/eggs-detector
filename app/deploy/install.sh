#!/usr/bin/env bash
# Instala o actualiza la app en una instancia Ubuntu (probado en EC2 con Ubuntu 24.04).
#
#   git clone https://github.com/dportilla219/eggs-detector.git ~/eggs-detector
#   # copiar el split de test (images/ + labels/) a ~/eggs-data/test  (opcional, para la banda)
#   bash ~/eggs-detector/app/deploy/install.sh
#
# Grupo de seguridad: abrir TCP 8000 (HTTP directo) y TCP 80 + 443 (HTTPS con Caddy).
set -euo pipefail

REPO="$HOME/eggs-detector"
DATA="$HOME/eggs-data"
DEPLOY="$REPO/app/deploy"

sudo apt-get install -y -qq python3-venv caddy >/dev/null
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$REPO/app/requirements.txt"

# Evaluación de la tubería completa sobre el test (la muestra la pestaña de métricas).
# Se repite si cambia el detector configurado en el servicio (p. ej. al pasar de v2 a v3).
DET=$(sed -n 's/^Environment=EGGS_DET_FILE=//p' "$DEPLOY/eggs-detector.service")
if [ -d "$DATA/test/images" ] && ! grep -qs "\"$DET\"" "$DATA/eval_test.json"; then
  (cd "$REPO/app/server" && "$REPO/.venv/bin/python" evaluate.py --split "$DATA/test" --out "$DATA/eval_test.json" --det "$DET")
fi

# App (puerto 8000)
sudo cp "$DEPLOY/eggs-detector.service" /etc/systemd/system/eggs-detector.service

# HTTPS con Caddy: el nombre <ip>.sslip.io se ajusta solo a la IP pública en cada arranque
sudo install -m 755 "$DEPLOY/caddy-host.sh" /usr/local/bin/eggs-caddy-host.sh
sudo mkdir -p /etc/systemd/system/caddy.service.d
printf '[Service]\nExecStartPre=/usr/local/bin/eggs-caddy-host.sh\n' | sudo tee /etc/systemd/system/caddy.service.d/eggs.conf >/dev/null
sudo cp "$DEPLOY/Caddyfile" /etc/caddy/Caddyfile

sudo systemctl daemon-reload
sudo systemctl enable -q eggs-detector.service caddy.service
sudo systemctl restart eggs-detector.service caddy.service
sleep 4
curl -fsS http://127.0.0.1:8000/api/health && echo
grep -m1 -oE '^[0-9-]+\.sslip\.io' /etc/caddy/Caddyfile | sed 's|^|HTTPS: https://|'
