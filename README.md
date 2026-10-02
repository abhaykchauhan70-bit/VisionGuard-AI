# VisionGuard AI — Real-Time Video Intelligence & Human Activity Understanding

A full-stack video intelligence system: FastAPI + MySQL backend, JWT auth,
OpenCV video processing, YOLO object detection, and a CNN-LSTM hybrid deep
learning pipeline for activity recognition with Grad-CAM explainability.

> **Honesty note:** the deep-learning pipeline (CNN feature extractor, YOLO
> detector, LSTM/GRU, hybrid model, Grad-CAM) is real, working code. YOLO
> object detection works immediately (pretrained on COCO). The **activity
> classifier (CNN-LSTM) needs to be trained on a real dataset you provide**
> — this project does not ship fabricated accuracy numbers. Until you train
> it, `/videos/{id}/analyze` will still run YOLO detection and save it, but
> the activity-prediction section will say "model not trained yet."

---

## 1. Project structure

```
VisionGuard-AI/
├── backend/            FastAPI app: routes, services, DB models, auth
├── ai/                 Framework-agnostic ML code: models, training, inference
├── frontend/            Plain HTML/CSS/JS dashboard (no build step needed)
├── sql/schema.sql       Reference MySQL schema (also auto-created by SQLAlchemy)
├── uploads/              Uploaded raw videos (gitignored)
├── results/              Processed videos with bounding boxes (gitignored)
├── models/               Trained model checkpoints (gitignored)
├── tests/                Pytest tests
├── requirements.txt
├── .env.example
├── Dockerfile / docker-compose.yml
```

## 2. Prerequisites (Windows)

- Python 3.11+ (https://www.python.org/downloads/)
- MySQL 8.0 (https://dev.mysql.com/downloads/installer/) OR use Docker (step 7)
- VS Code with the Python extension
- Git (optional)

## 3. Setup — run these in VS Code's terminal, from the project root

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (PowerShell)
venv\Scripts\activate

# If PowerShell blocks the script, run this once first:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3. Install dependencies (this also pulls PyTorch/OpenCV — may take a few minutes)
pip install -r requirements.txt
```

## 4. Configure the database

1. Create a MySQL database (if not using Docker):
   ```sql
   CREATE DATABASE visionguard;
   ```
2. Copy `.env.example` to `.env` and fill in your MySQL password:
   ```powershell
   copy .env.example .env
   ```
   Edit `.env` and set `DB_PASSWORD` and a random `JWT_SECRET_KEY`.

Tables are created automatically the first time the backend starts
(`init_db()` in `backend/app.py`). `sql/schema.sql` is provided as a
readable reference / for manual setup if you prefer.

## 5. Run the backend

```powershell
uvicorn backend.app:app --reload
```

- API root: http://127.0.0.1:8000
- Swagger docs (test every endpoint here): http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Leave this terminal running.

## 6. Run the frontend

The frontend is plain HTML/CSS/JS — no build step. Easiest option: install
the **Live Server** extension in VS Code, right-click `frontend/index.html`
→ "Open with Live Server". It will open at something like
`http://127.0.0.1:5500/frontend/index.html`.

(If Live Server uses a different port, add it to `FRONTEND_ORIGINS` in `.env`.)

Register an account, log in, and upload a short MP4 to try the pipeline —
YOLO detection + bounding-box video will run immediately.

## 7. Run everything with Docker instead (optional)

```powershell
docker compose up --build
```
This starts MySQL + the backend together. Still open `frontend/index.html`
with Live Server as in step 6 (the frontend itself isn't containerized).

## 8. Training the activity-recognition model (CNN-LSTM)

1. Gather labelled video clips into:
   ```
   ai/datasets/train/<activity_name>/clip1.mp4, clip2.mp4, ...
   ai/datasets/val/<activity_name>/...
   ai/datasets/test/<activity_name>/...
   ```
   Good public sources: **UCF101**, **HMDB51**, or a small set of clips you
   record yourself (e.g. walking / running / sitting / falling).
2. Train the hybrid model:
   ```powershell
   python -m ai.training.train_hybrid --epochs 20 --seq_len 8 --batch_size 4
   ```
   This prints real per-epoch accuracy/F1 and saves the best checkpoint to
   `models/cnn_lstm_best.pt`. Once this file exists, `/videos/{id}/analyze`
   will include a real activity prediction.
3. For the model-comparison experiment (Phase 17 requirement), also run:
   ```powershell
   python -m ai.training.train_cnn
   python -m ai.training.train_lstm --cell_type LSTM
   python -m ai.training.train_lstm --cell_type GRU
   ```
   Log each run's final accuracy/precision/recall/F1 into the
   `model_metrics` table (insert manually via `/docs` or write a small
   script) — the Model Performance page reads real rows from that table.

## 9. Testing

```powershell
pytest -v
```

## 10. Common errors

| Error | Fix |
|---|---|
| `Can't connect to MySQL server` | Make sure MySQL is running and `.env` credentials are correct |
| `ModuleNotFoundError: No module named 'backend'` | Run uvicorn from the **project root**, not inside `backend/` |
| CORS error in browser console | Add your Live Server URL to `FRONTEND_ORIGINS` in `.env` and restart uvicorn |
| YOLO download fails (no internet) | `ultralytics` needs internet once to download `yolov8n.pt`; after that it's cached |
| `No trained checkpoint found` | Expected until you complete step 8 — YOLO detection still works without it |

## 11. Explaining this project in an interview

- **Transfer learning**: ResNet18 pretrained on ImageNet is reused as a
  frozen feature extractor, so the LSTM only needs to learn temporal
  patterns, not visual features from scratch.
- **Why LSTM over a single CNN**: activities are defined by frame *order*,
  not any one frame — LSTM/GRU carries state across the sequence.
- **Attention**: instead of trusting only the last LSTM timestep, an
  attention layer learns which frames mattered most for the decision.
- **Grad-CAM**: explains *where* the CNN looked, via gradients flowing back
  to the last conv layer — a standard explainability technique for CNNs.
- **Security**: bcrypt password hashing, JWT auth on every protected route,
  file-type/size validation, filename sanitization (UUID renaming), CORS
  allow-list, secrets only via `.env`.
