import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Check for TensorFlow/Keras availability
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
except ImportError:
    st.error("TensorFlow is required to run this app. Please run: pip install tensorflow")
    st.stop()

# Page configuration
st.set_page_config(page_title="Bitcoin Multi-Horizon Forecaster", page_icon="🪙", layout="wide")

st.title("🪙 Bitcoin Multi-Horizon Forecasting System")
st.markdown("This dashboard trains a **1D-CNN architecture** to predict future Bitcoin prices at **1-day, 3-day, and 7-day horizons** simultaneously using historic sequence windows.")

# --- SIDEBAR: DATA INPUT & CONFIGURATION ---
st.sidebar.header("📥 Data & Model Parameters")
uploaded_file = st.sidebar.file_uploader("Upload bitcoin.csv", type=["csv"])

# Hyperparameters
window_size = st.sidebar.slider("Lookback Window Size (Days)", min_value=7, max_value=60, value=14)
train_split = st.sidebar.slider("Train Split Ratio", min_value=0.5, max_value=0.9, value=0.8)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=100, value=20)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)


# Helper function to parse strings with financial suffixes like K, M, B, and %
def clean_vol_change(df_col):
    if isinstance(df_col, str):
        df_col = df_col.strip()
        if 'K' in df_col:
            return float(df_col.replace('K', '')) * 1000
        elif 'M' in df_col:
            return float(df_col.replace('M', '')) * 1000000
        elif 'B' in df_col:
            return float(df_col.replace('B', '')) * 1000000000
        elif '%' in df_col:
            return float(df_col.replace('%', ''))
        else:
            try:
                return float(df_col.replace(',', ''))
            except ValueError:
                return None
    return df_col


def load_and_preprocess(file, target_col='Price', date_col='Date'):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df_sorted = df.sort_values(by='Date').reset_index(drop=True)
   
    # Clean and convert core market pricing series fields to true floating-point format
    for col in ['Price', 'Open', 'High', 'Low']:
        if col in df_sorted.columns:
            # Force conversion to string, strip commas/spaces, and force to numeric
            df_sorted[col] = (
                df_sorted[col]
                .astype(str)
                .str.replace(r'[$\s,]', '', regex=True)
            )
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')
    
    # Process volumes and change percentages through text parser mapping
    if 'Vol.' in df_sorted.columns:
        df_sorted['Vol.'] = df_sorted['Vol.'].apply(clean_vol_change)
    if 'Change %' in df_sorted.columns:
        df_sorted['Change %'] = df_sorted['Change %'].apply(clean_vol_change)
        
    # Handle missing or coerced NaN values globally safely across target column
    df_sorted[target_col] = df_sorted[target_col].ffill().bfill()
    
    return df_sorted, target_col, date_col

    

def create_datasets(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback - 7 + 1):
        X.append(series[i : i + lookback])
        y.append([
            series[i + lookback],       # 1-day forecast
            series[i + lookback + 2],   # 3-day forecast
            series[i + lookback + 6]    # 7-day forecast
        ])
    return np.array(X), np.array(y)


# --- PIPELINE EXECUTION ---
if uploaded_file is not None:
    # 1. Process dataset
    df, target_column, date_column = load_and_preprocess(uploaded_file)
    st.success(f"Successfully loaded data! Target forecast metric: `{target_column}` using timeline: `{date_column}`")
    
    # --- DATE SELECTION INPUT FIELDS ---
    min_date = df[date_column].min().to_pydatetime()
    max_date = df[date_column].max().to_pydatetime()
    
    st.sidebar.subheader("📅 Target Evaluation Anchor")
    selected_date = st.sidebar.date_input(
        "Choose Baseline Forecast Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Show Preview
    with st.expander("👀 View Raw Processed Dataset Preview"):
        st.dataframe(df, use_container_width=True)
        
    # Extract structural series array
    series_data = df[target_column].to_numpy().reshape(-1, 1)

    
    # 2. Fit Scaling Metrics
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series_data).flatten()
    
    # Create multi-horizon structural arrays
    X, y = create_datasets(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # 3. Train-Test Time-Based Split (No Shuffling)
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    st.info(f"📊 Training Samples: **{len(X_train)}** | Evaluation Verification Samples: **{len(X_test)}**")
    
    # --- MODEL CONSTRUCTION & EXECUTION ---
    if st.button("🚀 Train 1D-CNN Multi-Horizon Engine"):
        with st.spinner("Compiling structural weights and training model..."):
            
            # Construct 1D Convolutional Sequence Model
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(3)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Fit weights
            model.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(X_test, y_test), 
                verbose=0
            )
            
            st.success("🎉 Neural Network Training Cycle Finalized!")
            
            # --- EVALUATION AND INVERSE TRANSFORMATION ---
            predictions = model.predict(X_test)
            
            y_test_real = np.zeros_like(y_test)
            preds_real = np.zeros_like(predictions)
            
            for horizon_idx in range(3):
                y_test_real[:, horizon_idx] = scaler.inverse_transform(y_test[:, horizon_idx].reshape(-1, 1)).flatten()
                preds_real[:, horizon_idx] = scaler.inverse_transform(predictions[:, horizon_idx].reshape(-1, 1)).flatten()
                
            # Compute overall evaluation summaries
            mae_1d = np.mean(np.abs(y_test_real[:, 0] - preds_real[:, 0]))
            mae_3d = np.mean(np.abs(y_test_real[:, 1] - preds_real[:, 1]))
            mae_7d = np.mean(np.abs(y_test_real[:, 2] - preds_real[:, 2]))
            
            # # Display Overall Dashboard Metrics Cards
            # st.subheader("📉 Test Dataset Performance Evaluation (MAE)")
            # m1, m2, m3 = st.columns(3)
            # m1.metric("1-Day Horizon MAE", f"${mae_1d:,.2f}")
            # m2.metric("3-Day Horizon MAE", f"${mae_3d:,.2f}")
            # m3.metric("7-Day Horizon MAE", f"${mae_7d:,.2f}")
            
            # --- CUSTOM DATE PREDICTION LOOKUP BLOCK ---
            st.subheader(f"🔮 Custom Horizon Forecasts From Baseline: {selected_date.strftime('%Y-%m-%d')}")
            
            # Find index matching user selected calendar date
            target_dt = pd.to_datetime(selected_date)
            date_indices = df[df[date_column] == target_dt].index
            
            if len(date_indices) == 0:
                st.error("Selected date is missing from database timeline parameters.")
            else:
                idx = date_indices[0]  # Safely extract the integer index value
                
                # Verify lookback requirements exist in matrix history
                if idx < (window_size - 1):
                    st.error(f"Insufficient historical lookback data. Select a date at least {window_size} days past dataset start.")
                else:
                    # Get baseline price for chosen date to calculate directional delta
                    baseline_scaled = scaled_data[idx]
                    baseline_price = scaler.inverse_transform([[baseline_scaled]])[0, 0]
                    
                    # --- NEW: DISPLAY BASELINE PRICE HERE ---
                    st.metric(
                        label=f"💰 Bitcoin Baseline Price ({selected_date.strftime('%Y-%m-%d')})", 
                        value=f"${baseline_price:,.2f}"
                    )
                    st.markdown("---") # Optional visual separator line
                    
                    # Isolate exact window features
                    custom_window = scaled_data[idx - window_size + 1 : idx + 1]
                    custom_input = custom_window.reshape((1, window_size, 1))
                    
                    # Generate multi-horizon node forecasts
                    raw_pred = model.predict(custom_input)
                    
                    # Inverse scaling back to standard dollar pricing dimensions
                    val_1d = scaler.inverse_transform([[raw_pred[0, 0]]])[0, 0]
                    val_3d = scaler.inverse_transform([[raw_pred[0, 1]]])[0, 0]
                    val_7d = scaler.inverse_transform([[raw_pred[0, 2]]])[0, 0]
                    
                    # Calculate metric variance relative to chosen baseline anchor
                    delta_1d = val_1d - baseline_price
                    delta_3d = val_3d - baseline_price
                    delta_7d = val_7d - baseline_price
                    
                    # Compute specific target forecast calendar days
                    date_1d = target_dt + pd.Timedelta(days=1)
                    date_3d = target_dt + pd.Timedelta(days=3)
                    date_7d = target_dt + pd.Timedelta(days=7)
                    
                    # Display explicit custom target horizon cards with direction indicators
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        label=f"📅 1-Day Out ({date_1d.strftime('%Y-%m-%d')})", 
                        value=f"${val_1d:,.2f}",
                        delta=f"${delta_1d:,.2f}" if delta_1d >= 0 else f"-${abs(delta_1d):,.2f}"
                    )
                    c2.metric(
                        label=f"📅 3-Days Out ({date_3d.strftime('%Y-%m-%d')})", 
                        value=f"${val_3d:,.2f}",
                        delta=f"${delta_3d:,.2f}" if delta_3d >= 0 else f"-${abs(delta_3d):,.2f}"
                    )
                    c3.metric(
                        label=f"📅 7-Days Out ({date_7d.strftime('%Y-%m-%d')})", 
                        value=f"${val_7d:,.2f}",
                        delta=f"${delta_7d:,.2f}" if delta_7d >= 0 else f"-${abs(delta_7d):,.2f}"
                    )
