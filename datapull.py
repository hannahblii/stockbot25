import yfinance as yf
import pandas as pd

def fetch_and_clean_stock_data(ticker, start_date, end_date, output_csv):
    """
    Fetches daily stock price data for a given ticker, cleans it, and saves to a CSV.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'BA').
        start_date (str): The start date for fetching data (YYYY-MM-DD).
        end_date (str): The end date for fetching data (YYYY-MM-DD).
        output_csv (str): The filename to save the cleaned data (e.g., 'ba_stock_data.csv').
    """
    try:
        # Fetch data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Basic Cleaning: Handle missing values (e.g., forward fill or drop)
        # For simplicity, let's drop rows with any missing values.
        cleaned_data = stock_data.dropna()

        # You might add more cleaning steps here based on your needs,
        # such as handling outliers, checking data types, etc.

        # Save to CSV
        cleaned_data.to_csv(output_csv)
        print(f"Successfully fetched, cleaned, and saved data for {ticker} to {output_csv}")

    except Exception as e:
        print(f"Error fetching or processing data for {ticker}: {e}")

# Define parameters
ticker_symbol = 'ba'
start = '2020-01-01'
end = '2024-12-31' # Use an end date in the future or current date
output_file = 'ba_stock_data.csv'

# Run the function
fetch_and_clean_stock_data(ticker_symbol, start, end, output_file)