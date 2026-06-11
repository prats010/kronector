# Kronector Technical Report: Machine Learning for Formula 1 Race Strategy

## 1. Abstract
Kronector is an end-to-end Machine Learning operations (MLOps) pipeline that predicts Formula 1 race winners. It leverages LightGBM for tabular prediction and a multi-agent Large Language Model (LLM) architecture to provide explainable, mathematically-backed insights. This report details the data acquisition, feature engineering, model selection, evaluation methodology, and production deployment mechanisms.

## 2. Data Acquisition & Preprocessing
The model ingests 12 years of Formula 1 telemetry (2014–2026).
- **FastF1 API:** Supplies granular, lap-by-lap telemetry, tire degradation, and compound choices from 2018 onwards.
- **Jolpica API:** Provides historical race results, driver standings, and constructor points for the 2014–2017 hybrid era.

### 2.1 Prevention of Data Leakage
Sports prediction models are highly susceptible to data leakage (e.g., using a driver's final championship position to predict race 1). We strictly enforce **chronological isolation**. Features like `championship_standing` and `driver_form` are calculated *prior* to the start of the predicted race. 

## 3. Feature Engineering
Predicting race winners requires decoding complex domain knowledge into tabular features.

### 3.1 Era Normalization
Formula 1 undergoes massive regulation changes (e.g., 2014 V6 Hybrids, 2022 Ground Effect). A 1:20.000 lap time in 2014 is fundamentally incomparable to a 1:20.000 in 2024.
To solve this, sector times and qualifying paces are **era-normalized** using Z-scores relative to the specific season's average pace, rather than absolute milliseconds.

### 3.2 Critical Encoded Features
- **Grid Position (`grid_position`):** The starting position. Historically, this is the highest predictive feature due to "dirty air" making overtaking difficult.
- **Driver Form (`driver_form_last3`):** An exponentially weighted moving average of the driver's points over the last 3 races.
- **Tire Degradation:** Modeled as the coefficient of lap time drop-off during long runs in Free Practice 2 (FP2).

## 4. Model Selection: Why LightGBM?
During experimentation, we evaluated Neural Networks, XGBoost, CatBoost, and LightGBM. LightGBM was selected for production for the following reasons:
1. **Handling of Categorical Variables:** LightGBM natively handles categorical variables (like `driver_name` and `team`) using an optimal split algorithm, significantly outperforming one-hot encoding which creates sparse matrices.
2. **Speed on Tabular Data:** F1 data is strictly tabular. Deep learning architectures (like MLPs or TabNet) failed to outperform Gradient Boosted Trees and took an order of magnitude longer to train.
3. **SHAP Compatibility:** LightGBM is fully supported by `shap.TreeExplainer`, which allows for millisecond-latency generation of explainability metrics in production.

## 5. Evaluation Methodology
Standard K-Fold cross-validation is invalid for temporal data. We utilized **TimeSeriesSplit (n=5)**. The model trains on seasons $T_0 \dots T_n$ and tests on $T_{n+1}$.

### 5.1 Baseline Comparison
To prove the model's validity, it must outperform simple heuristics.
- **Baseline 1 (Pole Position Wins):** Predicting the pole-sitter wins yields ~42% accuracy over the last decade.
- **Baseline 2 (Championship Leader Wins):** Predicting the current WDC leader yields ~51% accuracy.
- **Kronector LightGBM:** Achieves ~71% accuracy on out-of-sample data (2023-2025 seasons).

## 6. Multi-Agent LLM Architecture
A raw probability output (e.g., "0.45") is not actionable for a race strategist. We wrap the LightGBM inference engine in a 4-stage Agentic Pipeline (powered by Llama-3.3-70b):
1. **DataAgent:** Parses natural language into a strict JSON intent (resolving "Monaco 23" to `season: 2023, round: 6`).
2. **PredictionAgent:** Executes the LightGBM model and extracts SHAP values.
3. **CritiqueAgent:** A pure-Python deterministic safeguard. It rejects predictions where confidence is below 20% (statistical noise in a 20-car field).
4. **SynthesisAgent:** Translates the math and SHAP values into a natural language response formatted as a radio message.

## 7. Automated Drift Detection & MLOps
Formula 1 is a dynamic environment. A model trained on 2022 data will rapidly decay in 2024 as team hierarchies shift.
We implemented **Evidently AI** to calculate the Population Stability Index (PSI) of incoming telemetry against the training distribution.
If $PSI > 0.2$ on critical features (e.g., a midfield team suddenly becomes the fastest car), the system automatically triggers an MLflow hyperparameter tuning pipeline to retrain and hot-swap the model in production.

## 8. Limitations & Future Work
The current architecture cannot account for chaotic, non-deterministic events:
- **Safety Cars / Red Flags:** Highly unpredictable events that reset field gaps.
- **Mechanical Failures (DNFs):** Random reliability issues.
- **Weather Chaos:** Sudden rain dramatically alters tire strategy and grip models.

Future iterations will ingest live radar weather data and historical Safety Car probability distributions per track to improve confidence intervals.
