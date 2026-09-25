#!/usr/bin/env bash
# Se ejecuta antes de arrancar Caddy (ExecStartPre). Si la instancia cambió de IP pública
# (pasa al detener y encender un laboratorio de AWS sin IP elástica), pone en el Caddyfile
# el nombre <ip-con-guiones>.sslip.io de la IP nueva; Caddy pide solo el certificado nuevo.
set -u
CADDYFILE=/etc/caddy/Caddyfile
MD=http://169.254.169.254/latest
TOKEN=$(curl -s -m 5 -X PUT "$MD/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
IP=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $TOKEN" "$MD/meta-data/public-ipv4" || true)
if [[ ! "$IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  exit 0  # sin IP pública o sin metadatos: deja el Caddyfile como está
fi
HOST="${IP//./-}.sslip.io"
if ! grep -q "^$HOST {" "$CADDYFILE"; then
  sed -i -E "s/^[0-9]+-[0-9]+-[0-9]+-[0-9]+\.sslip\.io \{/$HOST {/" "$CADDYFILE"
  echo "Caddyfile actualizado a $HOST"
fi
