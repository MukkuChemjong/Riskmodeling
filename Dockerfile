# ── Step 1: Choose your base image ──────────────────────────────────────────
# This gives you a clean Python 3.11 environment
# "slim" strips out tools you don't need, keeping the image small (~130MB vs ~900MB)
FROM python:3.11-slim


RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# ── Step 2: Set the working directory inside the container ──────────────────
# All subsequent commands run from this folder
WORKDIR /app

# ── Step 3: Copy requirements FIRST (before your code) ──────────────────────
# Docker caches each step as a "layer"
# If requirements.txt hasn't changed, Docker skips reinstalling — much faster rebuilds
COPY requirements.txt .

# ── Step 4: Install dependencies ────────────────────────────────────────────
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 5: Copy your application code and model ────────────────────────────
COPY main.py .
COPY train_model.py .
COPY data.csv .
RUN python train_model.py

# ── Step 6: Security — never run as root inside a container ─────────────────
RUN useradd -m apiuser && chown -R apiuser /app
USER apiuser

# ── Step 7: Tell Docker which port your API listens on ──────────────────────
# This is documentation — it does not actually open the port
EXPOSE 8080

# ── Step 8: Health check ─────────────────────────────────────────────────────
# Docker pings this every 30 seconds
# If it fails 3 times in a row, Docker marks the container as unhealthy
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# ── Step 9: The command that starts your API ─────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]