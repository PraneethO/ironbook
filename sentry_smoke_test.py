"""
Sentry smoke test — fires ~100 calls across all instrumented endpoints so the
dashboard populates with real transactions, spans, and a few errors.
"""
import json
import random
import time
import httpx

BASE = "http://localhost:8000/api"
client = httpx.Client(timeout=10)

def log(msg): print(f"  {msg}")

created_ids: list[str] = []

# ── 1. Health checks (30x) ───────────────────────────────────────────────────
print("\n[1/6] Health checks (30x)")
for i in range(30):
    r = client.get(f"{BASE}/health")
    log(f"GET /health → {r.status_code}  backend={r.json().get('reconstruction_backend')}")

# ── 2. Create projects (15x) ─────────────────────────────────────────────────
print("\n[2/6] Create projects (15x)")
for i in range(15):
    name = f"Smoke Test World {i+1}"
    r = client.post(f"{BASE}/projects", json={"name": name})
    if r.status_code == 200:
        pid = r.json()["id"]
        created_ids.append(pid)
        log(f"POST /projects → {r.status_code}  id={pid[:8]}…")
    else:
        log(f"POST /projects → {r.status_code}")

# ── 3. List + fetch projects (20x) ───────────────────────────────────────────
print("\n[3/6] List + fetch projects (20x)")
for i in range(10):
    r = client.get(f"{BASE}/projects")
    log(f"GET /projects → {r.status_code}  count={len(r.json())}")

for pid in (created_ids[:10] if created_ids else []):
    r = client.get(f"{BASE}/projects/{pid}")
    log(f"GET /projects/{pid[:8]}… → {r.status_code}")

# ── 4. Agent calls — no ANTHROPIC_API_KEY so these return the friendly fallback
#       message, but they still hit the endpoint and fire Sentry spans + Langfuse
#       traces. (20x) ──────────────────────────────────────────────────────────
print("\n[4/6] Agent /act calls (20x)")
sample_messages = [
    "What is in front of me?",
    "Go to the red object",
    "Highlight the door",
    "Zoom in on the window",
    "Rotate clockwise slowly",
    "What does this room look like?",
    "Find the table",
    "Move forward two steps",
    "Reset the view",
    "Make the background black",
]
camera = {
    "mode": "orbit",
    "fov": 1.0,
    "eye": [0, 1, 3],
    "target": [0, 0, 0],
    "bounds": {"min": [-2, -2, -2], "max": [2, 2, 2]},
}
for i in range(20):
    msg = random.choice(sample_messages)
    body = {
        "message": msg,
        "camera": camera,
        "history": [],
    }
    r = client.post(f"{BASE}/agent/act", json=body)
    data = r.json()
    log(f"POST /agent/act [{i+1}] → {r.status_code}  answer={data.get('answer','')[:60]!r}")
    time.sleep(0.1)  # slight spacing so Sentry doesn't batch too aggressively

# ── 5. Intentional 404s to generate error events (10x) ──────────────────────
print("\n[5/6] Intentional 404s (10x)")
for i in range(10):
    fake_id = f"nonexistent-project-{i}"
    r = client.get(f"{BASE}/projects/{fake_id}")
    log(f"GET /projects/{fake_id[:20]} → {r.status_code}")

# ── 6. Reconstruct on a real project (triggers the job pipeline + Sentry spans)
print("\n[6/6] Trigger reconstructions (5x)")
for pid in (created_ids[:5] if created_ids else []):
    r = client.post(f"{BASE}/projects/{pid}/reconstruct")
    log(f"POST /projects/{pid[:8]}…/reconstruct → {r.status_code}  status={r.json().get('status')}")

# ── Flush Sentry before exiting ──────────────────────────────────────────────
print("\nFlushing Sentry events…")
try:
    import sentry_sdk
    sentry_sdk.flush(timeout=5)
    print("Flushed.")
except Exception as e:
    print(f"(flush skipped: {e})")

print(f"\nDone. Fired calls across 6 endpoint groups.")
print("Check: https://praneeth-otthi.sentry.io/performance/")
