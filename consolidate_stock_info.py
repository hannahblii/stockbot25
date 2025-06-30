import pandas as pd
import os # To check for file existence and manage paths

ticker_symbols = [
    "adbe", "amd", "amzn", "appl", "avgo", "ba"
    "crm", "csco", "google", "jnj", "jpm", "meta",
    "msft", "nflx", "nvda", "pltr", "shop",
    "tsla", "uber", "xom", 
]

# Directory where your CSV files are located
# If your CSVs are in the same folder as this script, you can leave it as '.'
# Otherwise, provide the correct path, e.g., 'data_files/' or '/Users/YourUser/Documents/stock_data/'
csv_directory = '.'

# Output filename for the consolidated data
output_filename = 'consolidated_all_stocks_data.csv'

# --- Data Processing and Consolidation ---
all_stocks_data = []

print("\n--- Processing and Consolidating CSV files ---")
for ticker in ticker_symbols:
    file_path = os.path.join(csv_directory, f"{ticker}_data_with_all_indicators.csv")

    if not os.path.exists(file_path):
        print(f"  WARNING: File not found for {ticker} at {file_path}. Skipping this stock.")
        continue

    try:
        # Read the CSV file
        # header=0 (default) assumes the first row is the header
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

        # Ensure 'close' column is lowercase for consistency if not already
        if 'Close' in df.columns and 'close' not in df.columns:
            df.rename(columns={'Close': 'close'}, inplace=True)
        elif 'Adj Close' in df.columns and 'close' not in df.columns:
            df.rename(columns={'Adj Close': 'close'}, inplace=True)
        # Add more `elif` conditions if your close price column has other names

        # Add the 'Ticker' column to identify the stock in the consolidated DataFrame
        df['Ticker'] = ticker

        # Append the processed DataFrame to our list
        all_stocks_data.append(df)
        print(f"  Successfully processed {ticker}. Shape: {df.shape}")

    except Exception as e:
        print(f"  ERROR processing {ticker} from {file_path}: {e}")

print("-" * 30)

# --- Consolidate into a single DataFrame ---
if not all_stocks_data:
    print("No data was successfully loaded. Cannot consolidate.")
else:
    # Concatenate all individual stock DataFrames
    # Create a MultiIndex with 'Ticker' as the first level and 'Date' as the second.
    # This is ideal for LSTM models as you can easily group by ticker.
    consolidated_df = pd.concat(all_stocks_data).set_index('Ticker', append=True).swaplevel(0, 1)

    # Sort the index for consistent ordering (Ticker then Date)
    consolidated_df.sort_index(inplace=True)

    print("\n--- Consolidated DataFrame Head (MultiIndex: Ticker, Date) ---")
    print(consolidated_df.head())

    print("\n--- Consolidated DataFrame Info (showing columns and non-nulls) ---")
    consolidated_df.info()

    # --- Handle NaNs from Technical Indicator Calculation ---
    # Technical indicators (especially those with longer lookback periods like SMA_200)
    # will have NaN values at the beginning of each stock's series.
    # For LSTM training, you generally need to drop these rows.
    initial_rows = len(consolidated_df)
    consolidated_df.dropna(inplace=True)
    rows_after_dropna = len(consolidated_df)

    print(f"\nNaNs removed: {initial_rows - rows_after_dropna} rows dropped.")
    print(f"Shape of DataFrame after dropping NaNs: {consolidated_df.shape}")

    print("\n--- Consolidated DataFrame Head (After NaN removal) ---")
    print(consolidated_df.head())
    print("\n--- Consolidated DataFrame Tail ---")
    print(consolidated_df.tail())
    print("-" * 30)

    # --- Save the Consolidated Data to a new CSV ---
    try:
        consolidated_df.to_csv(output_filename)
        print(f"\nConsolidated data saved successfully to '{output_filename}'")
    except Exception as e:
        print(f"ERROR saving consolidated data to {output_filename}: {e}")
