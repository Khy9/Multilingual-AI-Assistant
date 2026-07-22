# Deploying to AWS App Runner

App Runner builds and runs a container from your `Dockerfile`, gives you an HTTPS URL, and handles
scaling. It suits this project because the single container serves both the API and the UI.

> **Verification note:** the Docker build was not executed on the machine this project was built on
> (Docker was not installed there). Build the image locally with `docker compose up --build` before
> pushing, so you find any build problem on your laptop instead of in a cloud build log.

---

## Step 0 — Set up budget alerts BEFORE you deploy

Do this first. App Runner is **not** free — it bills for provisioned container memory even while
idle, and a service left running for a month is the classic way a student project turns into a
surprise bill.

1. AWS Console → **Billing and Cost Management** → **Budgets** → **Create budget**.
2. Choose **Zero spend budget** (alerts on the first cent) or a **Cost budget** with a monthly cap
   of a few dollars.
3. Add your email as an alert recipient. Set thresholds at 50%, 80% and 100% of the cap.
4. Also enable **Billing** → **Billing preferences** → *Receive Free Tier Usage Alerts*.

Budget alerts are informational — **they do not stop spending.** When you are done demoing, run
`aws apprunner delete-service` or delete the service in the console. Pausing still bills for
provisioned memory.

Separately, remember the Gemini side: keep billing **disabled** on the Google Cloud project your
API key belongs to, or you silently leave the Gemini free tier. See README.md.

---

## Step 1 — Push the code to GitHub

App Runner's source-based deployment reads the repository directly.

The repository is already initialised, so this is just committing any outstanding work
and adding a remote:

```bash
cd multilingual-ai-assistant
git status                 # commit anything outstanding first — App Runner builds
git add -A                 # what is on GitHub, not what is on your disk
git commit -m "Ready to deploy"
git branch -M main
git remote add origin https://github.com/<you>/multilingual-ai-assistant.git
git push -u origin main
```

**Before pushing, confirm no key is committed:**

```bash
# .env lives at the PROJECT ROOT (backend/.env also works), so match both.
git ls-files | grep -E '(^|/)\.env$' && echo "STOP: .env is tracked!" || echo ".env is not tracked."
git ls-files | xargs grep -l "AIza" 2>/dev/null || echo "No API keys in tracked files."

# Stronger: scan every blob in history, not just the current checkout.
git rev-list --all --objects | awk '{print $1}' | \
  while read s; do git cat-file blob "$s" 2>/dev/null | grep -qE "AIza[0-9A-Za-z_-]{20,}" \
  && echo "LEAK in object $s"; done; echo "history scan done"
```

`.env` is listed in `.gitignore`, but check anyway — if it was ever committed, removing it from the
working tree does not remove it from history, and you must rotate the key.

---

## Step 2 — Create the App Runner service

1. AWS Console → **App Runner** → **Create service**.
2. **Source**: *Source code repository* → connect GitHub → pick your repo and the `main` branch.
   Set **Deployment trigger** to *Manual* (automatic redeploys on every push cost money).
3. **Build settings**: choose **Configure all settings here** — *not* "Use a configuration
   file" (see the note below).
   - Runtime: **Dockerfile**
   - Dockerfile path: `backend/Dockerfile`
   - Source directory / build context: `/` (the repository root — the image needs both
     `backend/` and `frontend/`)
4. **Service settings**:
   - Port: **8000** — must match `EXPOSE 8000` and the `--port 8000` in the Dockerfile's
     CMD, or health checks never pass.
   - CPU / Memory: **0.25 vCPU / 0.5 GB** — the smallest option, and enough here.
   - Health check: protocol **HTTP** (the default is TCP), path `/health`. TCP only proves
     the port is open; HTTP proves FastAPI actually booted.
5. **Environment variables** — see the next step.
6. Create, and wait for the build. The first one takes 5–15 minutes: installing
   `chromadb` and `onnxruntime` is slow.

### Why there is no `apprunner.yaml`

`apprunner.yaml` configures App Runner's **managed runtimes** (python3, nodejs, …), where
App Runner installs dependencies and runs a start command for you. It is not the mechanism
for Dockerfile builds — there is no `runtime: docker` managed runtime. For a Dockerfile-based
source deployment you select **Configure all settings here** in the console and point at
`backend/Dockerfile`; no config file is involved. Adding one only causes confusion.

---

## Step 3 — Environment variables (this is the important part)

**`GEMINI_API_KEY` is set in App Runner's configuration, never in the image.** Nothing in
`backend/Dockerfile` copies a `.env` file or bakes a key into a layer — a key baked into an image is
readable by anyone who can pull it, and `docker history` will show it even if a later layer deletes
the file.

In **Service settings → Environment variables**:

| Key | Value | Notes |
|---|---|---|
| `GEMINI_API_KEY` | *your key* | **Required.** Prefer a Secrets Manager reference (below) over a plaintext value |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Optional — `config.py` already defaults to this. Not a secret |
| `EMBED_MODEL` | `gemini-embedding-001` | Optional — same default in code. Not a secret |

> **Do not set `GEMINI_MODEL=gemini-2.5-flash-lite`.** That generation is retired for
> newly-created API keys: it still appears in `client.models.list()` but returns
> `404 — no longer available to new users` at request time. It would deploy cleanly and
> then fail on the first message. See README.md §6.

`pydantic-settings` reads these from the process environment exactly as it reads a local `.env`, so
no code changes are needed between local and deployed runs.

### Better: store the key in Secrets Manager

```bash
aws secretsmanager create-secret \
  --name multilingual-assistant/gemini-api-key \
  --secret-string "YOUR_KEY_HERE"
```

Then in App Runner add the environment variable with source **Secrets Manager**, referencing that
secret's ARN, and grant the service's instance role `secretsmanager:GetSecretValue` on it. The key
is then never visible in the App Runner console, and rotating it does not require a redeploy.

---

## Step 4 — Verify the deployment

```bash
curl https://<your-service>.awsapprunner.com/health
# {"status":"ok","llm_configured":true,...}
```

`llm_configured: true` confirms the key reached the container. If it is `false`, the environment
variable did not apply — check for a typo in the variable name and redeploy.

Then open the URL in a browser and send a message. **Streaming is the thing to check in
production**: some proxies buffer responses, which would deliver the whole reply at once. The app
sends `X-Accel-Buffering: no` and `Cache-Control: no-cache` to prevent this. If replies appear all
at once, buffering is the cause.

---

## If the build or first start fails

| Symptom in the App Runner logs | Cause | Fix |
|---|---|---|
| `gcc: command not found`, or `Failed building wheel for chroma-hnswlib` | `python:3.11-slim` ships no compiler. `chroma-hnswlib` and `onnxruntime` are native extensions; they normally install from manylinux wheels, but a fallback to source needs a toolchain | Add before `pip install` in `backend/Dockerfile`:<br>`RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*` |
| `Dockerfile not found` | Wrong path | Dockerfile path must be exactly `backend/Dockerfile`, source directory `/` |
| Build succeeds, health check fails, service rolls back | Port mismatch, or the container was OOM-killed | Confirm port `8000`; if Application logs show a kill, move up to 0.5 vCPU / 1 GB |
| `404 ... no longer available to new users` on the first message | Retired model pinned via `GEMINI_MODEL` | Use `gemini-3.5-flash-lite`, or drop the variable and use the code default |
| `/health` returns `llm_configured: false` | The env var did not apply | Check the key name spelling under **Configuration → Edit**, save, redeploy |

---

## Persistence caveat

App Runner instances have **ephemeral storage**. The `data/` directory — Chroma vectors, semantic
cache, user memory — is wiped on every redeploy and every scale-in. Uploaded documents do not
survive.

For a demo that is fine, and worth stating in your writeup. To make it durable you would move the
vector store to a hosted service (Chroma Cloud, Pinecone, or pgvector on RDS) and the cache to
ElastiCache — both are single-module swaps, as described in ARCHITECTURE.md.

---

## Cost control checklist

- [ ] Budget alert created **before** deploying
- [ ] Smallest instance size (0.25 vCPU / 0.5 GB)
- [ ] Auto-deploy on push disabled
- [ ] Billing **disabled** on the Google Cloud project holding the Gemini key
- [ ] **Service deleted when the demo is over** — this is the one that actually saves money

```bash
aws apprunner list-services
aws apprunner delete-service --service-arn <arn>
```
