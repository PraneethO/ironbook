#!/usr/bin/env bash
#
# dev.sh — run the Gaussian Splat World frontend on this laptop and tunnel
# /api traffic to the FastAPI backend running on the GPU server.
#
#   Vite (localhost:5173)  ->  /api proxy  ->  SSH tunnel (localhost:8000)
#                                              ->  GPU server FastAPI (:8000)
#
# Start the backend on the GPU server FIRST, then run this script:
#   cd /home/looq/langdon/calhacks/ironbook/backend
#   venv_gpu/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
#
set -euo pipefail

SSH_TARGET="looq@107.219.110.89"
SSH_PORT="2223"
FORWARD="8000:localhost:8000"
LOCAL_PORT="8000"

# Run from the directory this script lives in (frontend/) so `npm run dev`
# picks up the right package.json regardless of where it was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TUNNEL_PID=""

cleanup() {
  # Disarm traps so this only runs once.
  trap - EXIT INT TERM
  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo
    echo "Closing SSH tunnel (PID $TUNNEL_PID)..."
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# 1. Open the SSH tunnel in the background.
#    -f  : background after authentication
#    -N  : no remote command, just forward
#    ExitOnForwardFailure=yes : fail (non-zero) instead of silently continuing
#                               if the local port can't be bound.
echo "Opening SSH tunnel: localhost:${LOCAL_PORT} -> ${SSH_TARGET}:${SSH_PORT} (${FORWARD})"
if ! ssh -f -N -o ExitOnForwardFailure=yes -p "$SSH_PORT" -L "$FORWARD" "$SSH_TARGET"; then
  echo "ERROR: Failed to open SSH tunnel to ${SSH_TARGET}:${SSH_PORT}." >&2
  echo "       Is localhost:${LOCAL_PORT} already in use, or is the GPU server unreachable?" >&2
  exit 1
fi

# ssh -f daemonizes, so capture the backgrounded process by its forward spec.
TUNNEL_PID="$(pgrep -n -f "ssh.*-L ${FORWARD} ${SSH_TARGET}" || true)"
if [[ -n "$TUNNEL_PID" ]]; then
  echo "SSH tunnel established (PID $TUNNEL_PID)."
else
  echo "WARNING: tunnel started but its PID could not be determined;" >&2
  echo "         you may need to close it manually later." >&2
fi

# 2. Wait up to 5 seconds for localhost:8000 to become reachable.
echo "Waiting for localhost:${LOCAL_PORT} to become reachable..."
reachable=0
for _ in $(seq 1 50); do
  if nc -z localhost "$LOCAL_PORT" 2>/dev/null; then
    reachable=1
    break
  fi
  sleep 0.1
done

if [[ "$reachable" -ne 1 ]]; then
  echo "ERROR: localhost:${LOCAL_PORT} did not become reachable within 5 seconds." >&2
  echo "       The tunnel is up but nothing is listening on the GPU server's port 8000." >&2
  echo "       Did you start the backend on the GPU server?" >&2
  echo "         cd /home/looq/langdon/calhacks/ironbook/backend" >&2
  echo "         venv_gpu/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000" >&2
  exit 1
fi
echo "localhost:${LOCAL_PORT} is reachable."

# 3. Run the Vite dev server. cleanup() kills the tunnel when this exits.
echo "Starting Vite dev server (npm run dev)..."
echo "Open http://localhost:5173 in your browser. Press Ctrl-C to stop."
npm run dev
