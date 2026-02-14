# EV Material Chasis Selector

Predict whether a material is suitable for EV chassis selection using a trained Gradient Boosting model and a React UI.

**Deployment:** https://ev-material-chasis-selector.vercel.app

## Preview:
<img width="1899" height="961" alt="Screenshot 2026-02-15 004021" src="https://github.com/user-attachments/assets/80b4fa19-697e-4683-a7d5-0047f6ec1a63" />

## Highlights
- Predicts suitability from mechanical properties (Su, Sy, E, G, mu, Ro).
- Shows a clear decision (Suitable / Not suitable) with probability when available.
- Flask API with health check and JSON prediction endpoint.

## Tech Stack
- Frontend: React (Create React App)
- Backend: Flask, scikit-learn, pandas
- Model: GradientBoostingClassifier persisted with joblib

## Project Structure
```
backend/
	app.py                    Flask API
	train_model.py            Model training script
	requirements.txt          Backend dependencies
	data/material.csv         Training dataset
	model/material_gbc.joblib Trained model bundle
frontend/
	src/                      React app
```

## API
Base URL: `http://127.0.0.1:5000`

### `GET /health`
Returns service status.

### `POST /predict`
Request body:
```json
{
	"Su": 450,
	"Sy": 275,
	"E": 198000,
	"G": 77000,
	"mu": 0.29,
	"Ro": 7820
}
```

Response:
```json
{
	"prediction": 1,
	"usable": true,
	"probability": 0.83
}
```

## Local Setup

### 1) Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

The API runs at `http://127.0.0.1:5000`.

Optional env vars:
- `MODEL_PATH` to point at a custom model bundle (default: `model/material_gbc.joblib`).

### 2) Frontend
```bash
cd frontend
npm install
npm start
```

The UI runs at `http://localhost:3000`.

Optional env vars:
- `REACT_APP_API_BASE` to point the UI at a deployed API (default: `http://127.0.0.1:5000`).

## Training Notes
The training dataset is stored at `backend/data/material.csv` and must include:
`Su`, `Sy`, `E`, `G`, `mu`, `Ro`, `Use`, `Material`.

`train_model.py` trains the classifier, prints metrics, and writes a bundle with:
- `model`: the fitted estimator
- `feature_columns`: expected input field order

## Deployment
- Frontend is deployed on Vercel: https://ev-material-chasis-selector.vercel.app/
- Deploy the backend separately (Render, Railway, or a VM) and set `REACT_APP_API_BASE`.

## License
Add your license details here.
