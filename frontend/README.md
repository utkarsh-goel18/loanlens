# LoanLens Frontend

React + Vite frontend for the LoanLens explainable AI credit decision simulator.

## Run locally

From the `frontend` directory:

```bash
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` by default.

## Backend

Start the LoanLens FastAPI backend separately:

```bash
uvicorn backend.app.main:app --reload
```

The frontend calls:

```text
POST http://127.0.0.1:8000/analyze
```

To use another backend URL, create `frontend/.env.local`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Screens

- `/` — Landing page
- `/home` — LoanLens overview/home
- `/apply` — 12-parameter application form
- `/decision` — live prediction + SHAP explanation
- `/what-if` — live counterfactual scenarios

The last successful analysis is stored in browser local storage so the Decision and What-If screens can share the same `/analyze` result.
