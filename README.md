# 🌍 CO₂ Emission Forecasting with MLOps

This project forecasts CO₂ emissions using real-time power system data from Denmark. It demonstrates a complete **MLOps pipeline** with automated data ingestion, model training, prediction, evaluation, and deployment using **GitHub Actions** and **Render**.


**🔗 [🌍 CO₂ Emission Forecasting Application ](https://co2-emission-render.onrender.com)**
---

## 🚀 Features

- 🔁 **Automated ETL Pipeline**: Fetches and processes real-time energy data every 6 hours.
- 🧠 **Forecasting Model**: Uses an Attention-based LSTM model to predict CO₂ emissions for the next 6 hours.
- 📊 **Streamlit Dashboard**: Visualizes live emissions, predictions, and model performance.
- 🛠️ **FastAPI Backend**: Serves data through secure REST APIs.
- 📦 **SQLite Database**: Stores historical data, model metadata, predictions, and evaluation logs.
- 🔄 **Scheduled CI/CD**: GitHub Actions triggers pipeline every 6 hours to update data and models.
- 🖥️ **Render Deployment**: Streamlit app and FastAPI backend deployed on Render.

---

## 🧱 Project Structure

```
app/
├── backend/                 # FastAPI server
├── database/                # SQLite DB location
├── frontend/                # Streamlit dashboard
├── models/                  # Trained PyTorch models and scalers
├── scripts/
│   ├── data_pipeline/       # ETL pipeline scripts
│   ├── model_pipeline/      # Training, evaluation, logging
│   └── prediction_pipeline/ # Prediction, retraining, accuracy check
├── worker/                  # Scheduler for 6-hour intervals
├── start.sh                 # Starts frontend and backend on Render
```

---

## 🔄 Workflow

### 1. ⛏ Data Pipeline
- Extracts power data from EnergiDataService API.
- Aggregates and engineers time-series features.
- Stores the cleaned data in `co2_emission.db`.

### 2. 🧠 Model Pipeline
- Loads engineered features.
- Trains an LSTM with attention mechanism.
- Saves the model and scalers.
- Logs metadata and metrics to the database.

### 3. 🔮 Prediction Pipeline
- Predicts next 6 hours of CO₂ emissions.
- Logs predictions into the database.
- Evaluates accuracy if actuals are available.
- Retrains model if accuracy < 65%.

### 4. 📅 GitHub Actions
- `.github/workflows/mlops_pipeline.yml` runs the pipeline:
  - Every 6 hours
  - On code push to `main`
  - Or via manual trigger
- Pushes updated DB, logs, and models.

---

## 📈 Streamlit Dashboard

Access the deployed dashboard:

**🔗 [co2-emission-render.onrender.com](https://co2-emission-render.onrender.com)**

**Tabs:**
- Last 24h CO₂ Emissions
- Next 6h Forecast
- Recent Predictions vs Actual
- Model Performance

---

## 🔧 Render Deployment

Configured using `render.yaml` with two services:

- **Streamlit Web App** (`co2-emission-app`)
- **Worker Service** (`co2-emission-worker`) runs `scheduler.py` every 6h

---

## 🧪 Local Development

### Install dependencies
```bash
cd app
pip install -r base-requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Run locally
```bash
# Create DB and run initial pipelines
python scripts/db_creation.py
python scripts/data_pipeline/d_pipeline.py
python scripts/prediction_pipeline/predict.py

# Launch backend
uvicorn backend.main:app --reload

# Launch frontend
streamlit run frontend/app.py
```

---

## 🧠 Model Details

| Model         | Architecture     | Input Window | Forecast Horizon |
|---------------|------------------|---------------|------------------|
| `LSTM_Attn`   | LSTM + Attention | 24 hours      | 6 hours          |

---

## 🗃️ Database Schema

- `aggregated_data`: Cleaned hourly data
- `engineered_features`: Lag/rolling stats
- `model_table`: Metadata about each trained model
- `model_training_sets`: Tracks training windows
- `predictions`: All predictions made by models
- `model_evaluations`: Evaluation metrics per model
- `new_data_log`: Data pipeline logs

---

## 🧠 Tech Stack

- **ML**: PyTorch, Scikit-learn
- **API**: FastAPI
- **Dashboard**: Streamlit, Plotly
- **Database**: SQLite
- **Automation**: GitHub Actions, Render Scheduler
- **Packaging**: Docker-ready and modular

---

## 🙌 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change or improve.

---

## 📄 License

MIT License
