# 🚕 NYC Taxi Fare Prediction — Deep Feedforward Neural Network

A beautiful dark-themed Streamlit web application that predicts New York City taxi fares using a Deep Feedforward Neural Network (MLP).

## Model Architecture

```
Input (33 features) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
```

- **Parameters:** 14,721 trainable weights
- **Optimizer:** Adam (lr = 0.0042)
- **Loss:** Mean Squared Error (MSE)
- **Training Data:** ~1.9 million NYC taxi trips (2009–2015)

## Test-Set Results (290,613 unseen trips)

| Metric | Value |
|---|---|
| MAE | $1.3962 |
| RMSE | $2.6249 |
| R² | 0.9192 |

The MLP ranks 2nd overall — just behind XGBoost (RMSE $2.57) and ahead of Random Forest (RMSE $2.75).

## Features

The app computes **33 engineered features** from 5 raw inputs:
- **Raw coordinates:** pickup/drop-off latitude & longitude, passenger count
- **Distance & bearing:** Haversine distance, Manhattan distance, bearing (sin/cos)
- **Temporal:** hour, day of week, month, cyclical encodings, weekend/night/rush-hour flags
- **Landmark distances:** distance to JFK, LaGuardia, Newark, Times Square
- **Airport flags:** pickup/drop-off/trip airport indicators

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select the repository, branch, and set main file to `app.py`

## Files

| File | Purpose |
|---|---|
| `app.py` | Dark-themed Streamlit dashboard with prediction UI, metrics, and model comparison |
| `predictor.py` | NumPy-only inference engine (no TensorFlow required at runtime) |
| `features.py` | Feature engineering pipeline (same code used during training) |
| `mlp_weights.npz` | Trained weights, input/output scaling parameters |
| `feature_config.json` | Feature list, hyperparameters, and test metrics |
| `final_mlp.keras` | Original Keras model (reference copy, not loaded by the app) |

## Limitations

- **Historical pricing** — fares are from 2009 to mid-2015
- **Straight-line distance** — uses Haversine, not actual driving routes
- **No traffic/tolls/tips** — these factors are unknown to the model
- Fares above ~$40 are under-predicted by ~$3 on average
