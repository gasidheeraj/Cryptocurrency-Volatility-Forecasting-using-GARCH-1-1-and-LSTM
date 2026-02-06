# Cryptocurrency Volatility Forecasting using GARCH and LSTM

## 📌 Project Overview
This project focuses on forecasting **cryptocurrency price volatility**, with Bitcoin (BTC) as the primary asset, using a combination of **statistical time-series models** and **deep learning approaches**.

### Why this matters (industry context)
- Volatility forecasting is critical for **risk management**, **portfolio allocation**, **derivatives pricing**, and **algorithmic trading**
- Accurate volatility estimates help financial institutions and trading firms manage exposure in highly volatile crypto markets
- Combines interpretable econometric models with modern neural networks, reflecting real-world quantitative workflows

### High-level approach
- Compute and analyze daily log-return-based volatility
- Establish statistical baselines and naive forecasts
- Model conditional volatility using GARCH-family models
- Enhance forecasting performance using LSTM-based neural networks
- Compare models using error-based evaluation metrics

---

## 🧠 Key Concepts & Techniques

### Time-Series Volatility Modeling
- Log returns
- Conditional heteroskedasticity
- Stationarity testing (Augmented Dickey-Fuller test)

### Statistical Models
- Random Walk / Naive baseline
- GARCH(1,1)
- Asymmetric volatility models (GJR-GARCH, TARCH)

### Deep Learning
- LSTM networks for sequence modeling
- Sliding window input reshaping
- Regularization using Dropout and Batch Normalization

### Evaluation Metrics
- RMSE (Root Mean Squared Error)
- RMSPE (Root Mean Squared Percentage Error)

### Feature Engineering
- Daily volatility from log returns
- Temporal aggregation and scaling (MinMaxScaler)

---

## 🏗️ Project Architecture

### Data Source & Ingestion
- Historical Bitcoin price data fetched using `yfinance`
- Time-indexed OHLC data transformed into log returns

### Data Preprocessing
- Stationarity checks using Augmented Dickey-Fuller test
- Volatility computation from returns
- Train / validation / test splits
- Scaling for neural network models

### Modeling Pipeline
- Baseline naive volatility forecast
- GARCH model for symmetric volatility estimation
- GJR-GARCH and TARCH for asymmetric shock response
- LSTM-based neural networks for non-linear temporal learning

### Evaluation Workflow
- Validation-based comparison across all models
- Error metrics logged consistently for fair comparison
- Training vs validation loss monitoring

### Visualization & Outputs
- Volatility time-series plots
- ACF/PACF diagnostics
- Training vs validation loss curves
- Model performance comparison plots

---

## ⚙️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Data & Math: `numpy`, `pandas`, `scipy`
- Visualization: `matplotlib`
- Time-Series Analysis: `statsmodels`, `arch`
- Machine Learning: `scikit-learn`
- Deep Learning: `TensorFlow`, `Keras`
- Data Ingestion: `yfinance`

### Tools
- Jupyter Notebook
- Local execution environment (CPU-based)

---

## 📊 Results & Performance
- Multiple volatility forecasting approaches were evaluated under a unified framework
- GARCH-family models effectively captured volatility clustering
- Asymmetric models improved responsiveness to negative return shocks
- LSTM models learned non-linear temporal dependencies
- Performance evaluated using **RMSE** and **RMSPE**
- Neural models required careful regularization to avoid overfitting

---

## 🔍 Key Engineering & Data Science Highlights
- End-to-end quantitative modeling pipeline from raw market data to evaluation
- Side-by-side comparison of econometric and deep learning approaches
- Clear separation between preprocessing, modeling, and evaluation stages
- Reproducible experimentation with consistent metrics
- Workflow aligned with industry quant, ML, and data science roles

---

## 🚀 Future Improvements
- Real-time volatility forecasting using live market APIs
- Production-grade pipeline with scheduled ingestion and retraining
- Multi-asset volatility modeling across cryptocurrencies
- Deployment-ready dashboards for volatility monitoring
- Cloud integration and workflow orchestration
- Hyperparameter optimization and model ensembling

---

## 📁 Repository Structure
## 🧪 How to Run
```bash
pip install numpy pandas matplotlib scipy statsmodels arch scikit-learn tensorflow yfinance
jupyter notebook
