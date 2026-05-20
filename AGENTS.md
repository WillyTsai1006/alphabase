# AlphaBase V3.0 — Agent Instructions

## Cursor Cloud specific instructions

### Overview

AlphaBase is a Python-based dual-AI quantitative trading system with a Streamlit dashboard. It consists of a single service (Streamlit) backed by a TimescaleDB (PostgreSQL 14) database running in Docker.

### Prerequisites (already installed by update script)

- Python 3.10+ with pip dependencies from `requirements.txt`
- Docker with `docker-compose-plugin` for TimescaleDB

### Starting services

1. **Docker daemon** (required first):
   ```bash
   sudo dockerd &> /tmp/dockerd.log &
   # Wait ~5s, then fix socket permissions:
   sudo chmod 666 /var/run/docker.sock
   ```

2. **TimescaleDB**:
   ```bash
   cd /workspace && docker compose up -d
   ```
   Credentials: user=`quant`, password=`password`, db=`alphabase`, port=5432.

3. **Data pipeline** (run from `/workspace/src`):
   ```bash
   python3 data_loader.py      # ETL: fetches stock data from Yahoo Finance
   python3 quant_engine.py     # Train primary LightGBM (Optuna, ~30s)
   python3 meta_engine.py      # Train Meta-Labeling model (~20s)
   python3 hmm_engine.py       # Train HMM market regime model (~2s)
   ```
   Model `.pkl` files are saved in `/workspace/src/`.

4. **Streamlit dashboard**:
   ```bash
   cd /workspace/src && streamlit run app.py --server.port 8501 --server.headless true
   ```

### Important caveats

- All Python scripts in `src/` use relative imports (`from config import ...`), so you must run them with cwd set to `/workspace/src`.
- The data pipeline must complete before the dashboard or backtester can function (they load `.pkl` model files from disk and query the DB).
- `docker compose` uses the compose plugin syntax (not `docker-compose` binary).
- Docker in this VM requires `fuse-overlayfs` storage driver and `iptables-legacy` due to nested container constraints. The daemon config is at `/etc/docker/daemon.json`.
- `~/.local/bin` must be on PATH for `streamlit` and other pip-installed CLI tools.
- There are no automated tests or linter configuration in this repository.
- Yahoo Finance data fetch requires outbound internet; after initial load, the DB is self-contained.
