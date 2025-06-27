import pandas as pd
import pandas_ta as ta

# --- Step 2: Read your CSV file into a Pandas DataFrame ---
try:
    df = pd.read_csv('pltr_stock_data.csv', index_col='Date', parse_dates=True)
    print("CSV file loaded successfully.")
    print(df.head())
    print("-" * 30)
except FileNotFoundError:
    print("Error: 'stock_data.csv' not found. Please make sure the file is in the same directory or provide the correct path.")
    exit()


# --- Step 3: Ensure you have a 'close' column (or similar) ---
# Check common column names for closing price
close_column_name = None
if 'Close' in df.columns:
    close_column_name = 'Close'
elif 'close' in df.columns: # pandas_ta prefers lowercase 'close'
    close_column_name = 'close'
elif 'Adj Close' in df.columns: # Common from yfinance
    close_column_name = 'Adj Close'
else:
    print("Error: No 'Close', 'close', or 'Adj Close' column found in the CSV.")
    print("Please ensure your CSV has a closing price column and adjust the code if its name is different.")
    exit()

# If your column name is 'Adj Close' or something else, it's good practice to rename it to 'close'
# for pandas_ta, or pass it directly. We'll rename it for simplicity.
if close_column_name != 'close':
    df.rename(columns={close_column_name: 'close'}, inplace=True)
    print(f"Renamed '{close_column_name}' column to 'close'.")
    print(df.head())
    print("-" * 30)

# --- Step 3: Compute RSI, MACD, and MA using pandas_ta ---

# RSI (Relative Strength Index)
# Default length is 14
df.ta.rsi(append=True)
print("RSI_14 calculated.")

# MACD (Moving Average Convergence Divergence)
# Default parameters are fast=12, slow=26, signal=9
df.ta.macd(append=True)
print("MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9 calculated.")

# Simple Moving Average (SMA)
# Common lengths are 50 and 200 for long-term trends
df.ta.sma(length=50, append=True) # 50-period SMA
df.ta.sma(length=200, append=True) # 200-period SMA (might be mostly NaN for short data)
print("SMA_50 and SMA_200 calculated.")

# --- Optional: Exponential Moving Average (EMA) ---
# Often preferred over SMA as they give more weight to recent prices
df.ta.ema(length=20, append=True) # 20-period EMA for short-term trend
print("EMA_20 calculated.")


print("\nIndicators calculated successfully. Here are the last few rows with new indicator columns:")
# Display a selection of the new indicator columns
print(df[['close', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'SMA_50', 'EMA_20']].tail(10)) # Showing last 10 rows

print(f"\nNumber of NaN values in RSI_14 column: {df['RSI_14'].isna().sum()}")
print(f"Number of NaN values in MACD_12_26_9 column: {df['MACD_12_26_9'].isna().sum()}")
print(f"Number of NaN values in SMA_50 column: {df['SMA_50'].isna().sum()}")
print(f"Number of NaN values in SMA_200 column: {df['SMA_200'].isna().sum()}")
print(f"Number of NaN values in EMA_20 column: {df['EMA_20'].isna().sum()}")


# Save the DataFrame with the new indicator columns to a new CSV if desired
df.to_csv('stock_data_with_all_indicators.csv')
print("\nDataFrame with all indicators saved to 'stock_data_with_all_indicators.csv'")