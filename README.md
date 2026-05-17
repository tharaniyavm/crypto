# CryptoCast

CryptoCast is a Streamlit-powered crypto forecasting project focused on Bitcoin price prediction using deep learning. The repository includes an interactive forecasting app, a model evaluation dashboard, and sample cryptocurrency price data.

## Project Structure

- `Home.py` - Main Streamlit landing page with project overview, multi-horizon forecasting explanation, and model/skill highlights.
- `pages/1_ModelComparison.py` - Streamlit page for comparing model performance metrics from `model_evaluation_comparison.csv`.
- `pages/2_BitcoinForecasting.py` - Streamlit forecasting page that trains a 1D-CNN model and predicts Bitcoin price outcomes for 1-day, 3-day, and 7-day horizons.
- `bitcoin.csv` - Sample Bitcoin historical data used by the forecasting page.
- `model_evaluation_comparison.csv` - Model performance comparison data used by the model evaluation dashboard.

## Features

- Multi-horizon Bitcoin forecasting using a 1D convolutional neural network.
- Interactive Streamlit interface for data upload, hyperparameter selection, and training execution.
- Forecast output for 1-day, 3-day, and 7-day horizons.
- Model comparison dashboard highlighting best-performing models by MAE.
- Data cleaning logic for prices, volumes, and percentage changes.

## Data Format

The `bitcoin.csv` dataset uses the following columns:

- `Date` - date formatted as `DD-MM-YYYY`
- `Price` - Bitcoin closing price
- `Open` - opening price
- `High` - intraday high price
- `Low` - intraday low price
- `Vol.` - traded volume (supports K/M/B suffixes)
- `Change %` - daily price change percentage

## Requirements

- Python 3.9+
- Streamlit
- pandas
- numpy
- scikit-learn
- tensorflow

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn tensorflow
```

## Run the App

From the repository root, run:

```bash
streamlit run Home.py
```

Then use the app navigation to access:

- `Model Evaluation Dashboard`
- `Bitcoin Multi-Horizon Forecasting System`

## Notes

- The forecasting page currently expects a valid `bitcoin.csv` file upload or a compatible dataset with the same columns.
- The 1D-CNN model is a simple sequence model designed for experimental forecasting and comparison.

## License

This repository does not include a license file. Add a license if you plan to publish or share the project widely.
