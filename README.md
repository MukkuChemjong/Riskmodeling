# Bankruptcy Risk Prediction API

> A machine learning API that predicts the probability of corporate bankruptcy based on financial features.

**Version:** 2.1.0  
**Base URL:** `https://bankruptcy-risk-api-clpmj5md2a-uc.a.run.app`  
**Deployed on:** Google Cloud Run  

---

## Table of Contents

- [Overview](#overview)
- [Target Users](#target-users)
- [Expected Daily Request Volume](#expected-daily-request-volume)
- [User Requirements](#user-requirements)
- [API Endpoints](#api-endpoints)
- [How to Interact with the Service](#how-to-interact-with-the-service)
- [Interpreting the Prediction Score](#interpreting-the-prediction-score)
- [Model Information](#model-information)

---

## Overview

This API exposes a trained Gradient Boosting machine learning model that predicts the probability of corporate bankruptcy based on financial features. It is deployed as a containerised FastAPI service on Google Cloud Run and returns real-time predictions via a simple REST interface.

---

## Target Users

The primary clients for this service are:

- **Credit analysts and risk officers** at banks or lending institutions who need a fast second opinion on a borrower's financial health before approving a loan.
- **Fintech and SaaS platforms** that embed credit risk scoring into their own dashboards or underwriting workflows.
- **Private equity and investment firms** performing due diligence on acquisition targets or portfolio companies.
- **Accounting and audit firms** that need an automated flag for going-concern risk during financial review.
- **Internal risk management teams** that want to monitor their existing loan books for early warning signals.

---

## Expected Daily Request Volume

| Deployment Tier | Expected Daily Requests | Use Case |
|---|---|---|
| Development / Testing | < 500 | Internal QA and integration testing |
| Small Firm / Pilot | 500 – 5,000 | Single analyst team or pilot client |
| Mid-size Platform | 5,000 – 50,000 | Embedded in a fintech product |
| Enterprise | 50,000+ | High-volume automated underwriting |

Google Cloud Run scales automatically from zero to handle traffic spikes. No pre-warming is required for typical workloads below 1,000 requests per minute.

---

## User Requirements

### Real-Time Responses
The API is optimised for synchronous, real-time scoring. Each request returns a prediction in under 200ms under normal load, making it suitable for interactive UIs, loan origination forms, or analyst dashboards.

### Batch Processing
The current version does not expose a native batch endpoint. To score multiple companies in bulk, send concurrent POST requests to `/predict`.

### Input Requirements
Callers must supply the same financial features the model was trained on. All values must be numeric (`float` or `int`). Missing values should be imputed to the column median before sending.

---

## API Endpoints

### `GET /health`

Returns the operational status of the service. Use for uptime monitoring and load-balancer health checks.

**Request**
```
GET /health
```

**Response — 200 OK**
```json
{
  "status": "ok"
}
```

---

### `POST /predict`

Accepts a JSON object of financial features and returns the probability that the company will go bankrupt.

**Request Headers**

| Header | Value | Required |
|---|---|---|
| `Content-Type` | `application/json` | Yes |
| `Accept` | `application/json` | Recommended |

**Request Body**

A flat JSON object where each key is a financial feature name and each value is a number.

```json
{
  "ROA(C) before interest and depreciation before interest": 0.043,
  "ROA(A) before interest and % after tax": 0.031,
  "Debt ratio %": 0.52,
  "Net worth/Assets": 0.48,
  "Borrowing dependency": 0.21,
  "Operating Gross Margin": 0.18
}
```

**Response — 200 OK**
```json
{
  "bankruptcy_probability": 0.1273
}
```

The returned value is a float between `0.0` and `1.0`. Values at or above `0.35` indicate elevated bankruptcy risk.

**Response — 422 Unprocessable Entity**

Returned when the request body fails validation.
```json
{
  "detail": [{ "loc": ["body"], "msg": "value is not a valid dict", "type": "type_error.dict" }]
}
```

---

### `GET /docs`

Interactive Swagger UI. Explore and test all endpoints directly in your browser without writing any code.

```
https://bankruptcy-risk-api-clpmj5md2a-uc.a.run.app/docs
```

---

## How to Interact with the Service

### Browser (No Code)

Open the interactive docs in any browser:

```
https://bankruptcy-risk-api-clpmj5md2a-uc.a.run.app/docs
```

Click **POST /predict → Try it out**, paste your JSON into the request body, and click **Execute**.

---


## Interpreting the Prediction Score

| Score Range | Risk Level | Recommended Action |
|---|---|---|
| 0.00 – 0.15 | 🟢 Low Risk | Standard processing. No additional review required. |
| 0.15 – 0.35 | 🟡 Moderate Risk | Flag for analyst review. Request additional documentation. |
| 0.35 – 0.60 | 🟠 High Risk | Escalate to senior credit officer. Enhanced due diligence. |
| 0.60 – 1.00 | 🔴 Critical Risk | Decline or require significant collateral and guarantees. |
