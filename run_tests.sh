#!/usr/bin/env bash
# Both suites plus a build, so a broken frontend or a broken backend is caught
# by the same command.
#
# The backend suite needs a Postgres it can create and drop databases in. If one
# is already reachable at TEST_DATABASE_ADMIN_URL it is used as-is; otherwise a
# throwaway container is started and stopped again.

set -euo pipefail

cd "$(dirname "$0")"

PY=.venv/bin/python
TEST_DB_URL="${TEST_DATABASE_ADMIN_URL:-postgresql://notebooklm:notebooklm@localhost:55432/notebooklm}"
STARTED_DB=""

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

cleanup() {
  if [ -n "$STARTED_DB" ]; then
    step "Stopping the scratch database"
    docker rm -f "$STARTED_DB" >/dev/null
  fi
}
trap cleanup EXIT

# Reachability check that works for either a local database or a container.
db_is_up() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import sys
import psycopg
from urllib.parse import urlparse

EOF
  "$PY" -c "
import sys, psycopg
psycopg.connect('''$TEST_DB_URL''', connect_timeout=3).close()
" >/dev/null 2>&1
}

if db_is_up; then
  step "Using the Postgres already running"
else
  step "Starting a scratch Postgres"
  STARTED_DB=notebooklm-test-db-$$
  docker run -d --rm --name "$STARTED_DB" \
    -e POSTGRES_USER=notebooklm \
    -e POSTGRES_PASSWORD=notebooklm \
    -e POSTGRES_DB=notebooklm \
    -p 55432:5432 pgvector/pgvector:pg16 >/dev/null

  for _ in $(seq 1 60); do
    db_is_up && break
    sleep 1
  done
  db_is_up || { echo "Postgres never became reachable on :55432" >&2; exit 1; }
fi

step "Backend: API, retrieval, sessions, migrations"
TEST_DATABASE_ADMIN_URL="$TEST_DB_URL" "$PY" test_app.py

step "Frontend: components and interaction"
(cd frontend && npm test)

step "Frontend: production build"
(cd frontend && npm run build)

step "All checks passed"