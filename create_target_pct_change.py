import pandas as pd
import numpy as np # Although not strictly needed for this specific calculation, good practice to import

# --- 1. Load your consolidated DataFrame ---
# IMPORTANT: Replace 'consolidated_all_stocks_data.csv' with the actual path
# to your consolidated CSV file generated in the previous step.
# Ensure it's loaded with the MultiIndex as you saved it.
try:
    consolidated_df = pd.read_csv(
        'consolidated_all_stocks_data.csv',
        index_col=['Ticker', 'Date'], # Specify the MultiIndex columns
        parse_dates=['Date']          # Ensure 'Date' is parsed as datetime
    )
    # Ensure the 'Date' level of the MultiIndex is indeed datetime objects
    # This is a good safeguard as read_csv might sometimes load it as a string initially
    # if it's part of index_col, depending on pandas version/data.
    consolidated_df.index = consolidated_df.index.set_levels(
        pd.to_datetime(consolidated_df.index.levels[1]), level='Date'
    )
    print("Consolidated DataFrame loaded successfully.")
    print(consolidated_df.head())
    print("\n")
except FileNotFoundError:
    print("Error: 'consolidated_all_stocks_data.csv' not found.")
    print("Please make sure you have run the consolidation script first, or provide the correct path.")
    exit()
except Exception as e:
    print(f"An error occurred loading the consolidated DataFrame: {e}")
    exit()


# --- 2. Prepare the 'close' column ---
# Ensure the 'close' column exists and is numeric.
# This handles potential issues if it was loaded as an object type.
if 'close' not in consolidated_df.columns:
    print("Error: 'close' column not found in the consolidated DataFrame.")
    print("Available columns:", consolidated_df.columns.tolist())
    exit()

consolidated_df['close'] = pd.to_numeric(consolidated_df['close'], errors='coerce')
# 'errors='coerce'' will turn any non-numeric values into NaN, which will then be dropped.


# --- 3. Generate the Target Variable: Next Day's Percentage Change ---

# Step 3.1: Get the next day's close price for each stock
# We use .groupby(level='Ticker') to ensure the shift operation is performed
# independently for each stock, preventing data leakage across tickers.
# .shift(-1) moves the values up by one row, so each row gets the *next* day's value.
consolidated_df['next_day_close'] = consolidated_df.groupby(level='Ticker')['close'].shift(-1)
print("Added 'next_day_close' column (temporary for calculation).")

# Step 3.2: Calculate the percentage change
# Formula: (Next Day's Close - Current Day's Close) / Current Day's Close * 100
consolidated_df['target_pct_change'] = (
    (consolidated_df['next_day_close'] - consolidated_df['close']) / consolidated_df['close'] * 100
)
print("Added 'target_pct_change' as the primary target variable.")


# --- 4. Handle NaNs introduced by Target Variable Creation ---
# The 'next_day_close' and 'target_pct_change' columns will have NaN values
# for the very last day of data for each stock, because there's no "next day" to shift from.
# We need to drop these rows as they cannot have a target.
print(f"\nBefore dropping target NaNs: {len(consolidated_df)} rows")
# Drop rows where the target_pct_change is NaN. This also covers cases where 'close' was NaN.
consolidated_df.dropna(subset=['target_pct_change'], inplace=True)
print(f"After dropping target NaNs: {len(consolidated_df)} rows")

# You can optionally drop the 'next_day_close' column now if you don't need it as a feature
consolidated_df.drop(columns=['next_day_close'], inplace=True)
print("Dropped 'next_day_close' column.")


# --- 5. Display Results and Save ---
print("\nConsolidated DataFrame with 'target_pct_change' (last few rows):")
# Display relevant columns to show the new target
print(consolidated_df[['close', 'target_pct_change']].tail())

print("\nInfo on the DataFrame after target generation:")
consolidated_df.info()

# Save the DataFrame with the new target variable
output_filename_with_targets = 'consolidated_data_with_pct_change_target.csv'
consolidated_df.to_csv(output_filename_with_targets)
print(f"\nDataFrame with percentage change target saved to '{output_filename_with_targets}'")