#!/bin/sh
# One-time setup of the demo GLPI started by itsm/docker-compose.yml. Safe to run again.
#
#   cd itsm && docker compose up -d && cd .. && scripts/setup_glpi.sh
#
# 1. waits until GLPI has installed its database (first start takes ~1 minute)
# 2. enables the REST API with login/password  (triago's poller and feedback export use it)
# 3. lets the API be called from the Docker host (demo only: never expose this GLPI)
# 4. installs and activates the triago plugin  (routes tickets while they are created)
# 5. creates triago's 9 assignment groups
set -eu

GLPI=${GLPI_CONTAINER:-itsm-glpi-1}
DB=${GLPI_DB_CONTAINER:-itsm-db-1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
sql() { docker exec "$DB" mysql -uglpi -pglpi glpi -N -e "$1" 2>/dev/null; }
console() { docker exec -u www-data "$GLPI" php bin/console "$@" -n; }

printf 'waiting for GLPI to finish installing'
for _ in $(seq 1 90); do
    [ -n "$(sql "SELECT value FROM glpi_configs WHERE name='version'" || true)" ] && break
    printf '.'; sleep 2
done
echo
[ -n "$(sql "SELECT value FROM glpi_configs WHERE name='version'" || true)" ] || { echo "GLPI is not ready: check 'docker logs $GLPI'"; exit 1; }

console config:set enable_api 1 >/dev/null
console config:set enable_api_login_credentials 1 >/dev/null
sql "UPDATE glpi_apiclients SET ipv4_range_start=NULL, ipv4_range_end=NULL WHERE name='full access from localhost'"
echo "API enabled (login/password, callable from the Docker host)"

console plugin:install triago -u glpi >/dev/null
console plugin:activate triago >/dev/null
echo "triago plugin installed and active"

PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY=python3
"$PY" -c "from triago.glpi import GLPI; GLPI().ensure_groups()" && echo "assignment groups created"
echo "done: GLPI at ${GLPI_URL:-http://localhost:8080/apirest.php}" | sed 's#/apirest.php##'
