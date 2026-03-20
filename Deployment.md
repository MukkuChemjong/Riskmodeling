# Bankruptcy Risk Detection API

## Local Development

```bash
git clone https://github.com/username/bankruptcy-risk-api.git
cd bankruptcy-risk-api
python -m venv venv

pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Docker

```bash
# Build
docker build -t bankruptcy-api:v2.1.0 .

# Run
docker run -p 8080:8080 bankruptcy-api:v2.1.0

# Run in background
docker run -d -p 8080:8080 --name risk-api bankruptcy-api:v2.1.0

# Verify
curl http://localhost:8080/api/v1/health

# Logs
docker logs -f risk-api

# Stop and remove
docker stop risk-api
docker rm risk-api
```

---

## GCP Cloud Run

```bash
# Authenticate
gcloud auth login
gcloud config set project PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create risk-api \
  --repository-format=docker \
  --location=us-central1 \
  --description="Bankruptcy Risk API container images"

# Configure Docker to authenticate with GCP
gcloud auth configure-docker us-central1-docker.pkg.dev

# Tag the image
docker tag bankruptcy-api:v2.1.0 \
  us-central1-docker.pkg.dev/PROJECT_ID/risk-api/bankruptcy-api:v2.1.0

# Push to Artifact Registry
docker push \
  us-central1-docker.pkg.dev/PROJECT_ID/risk-api/bankruptcy-api:v2.1.0

# Deploy to Cloud Run
gcloud run deploy bankruptcy-risk-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/risk-api/bankruptcy-api:v2.1.0 \
  --region us-central1 \
  --platform managed \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 10 \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars MODEL_VERSION=v2.1.0

# Get the live URL
gcloud run services describe bankruptcy-risk-api \
  --region us-central1 \
  --format "value(status.url)"

# Verify
curl https://CLOUD_RUN_URL/api/v1/health
```

---

## Dashboard

```bash
python -m http.server 8000
# Open http://localhost:8000/dashboard.html
```
