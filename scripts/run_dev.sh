#!/usr/bin/env bash
# Start the PackGuard pipeline service in dev mode.
# Port 8001 (Person 3 will use 8002, Person 4 frontend 3000).

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d ".venv" ]]; then
  echo ">> creating .venv"
  python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

pip install -q -r requirements.txt

exec uvicorn packguard_pipeline.main:app --reload --port 8001 --host 0.0.0.0
