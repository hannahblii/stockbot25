import pandas as pd
import numpy as np
from datetime import date, timedelta
import asyncio
import aiohttp
import json
import os
import time # For time.sleep

# Install transformers and torch if not already present
# In Google Colab, you might need to run:
# !pip install transformers torch aiohttp

# --- IMPORTANT: Patch asyncio for Colab/Jupyter environments ---
# This allows nested event loops, often resolving issues where the code
# stops without apparent error after initial setup.
import nest_asyncio
nest_asyncio.apply()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Configuration ---
TICKERS = ["adbe", "amd", "amzn", "appl", "avgo",
     "csco", "google", "jnj", "jpm", "meta",
    "msft", "nflx", "nvda", "pltr", "shop",
    "tsla", "uber", "xom",]
START_DATE = date(2024, 7, 9)
END_DATE = date(2024, 12, 31)
HEADLINES_PER_WEEK = 3 # User requested 3 headlines per week
OUTPUT_CSV_FILE = 'stock_sentiment_finnhub_finbert.csv'

# --- IMPORTANT: Replace with your actual Finnhub API Key ---
# A paid Finnhub plan is required for historical news data beyond 30 days.
FINNHUB_API_KEY = "d1muh59r01qlvnp4kukgd1muh59r01qlvnp4kul0"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1/"

# Sentiment mapping for numerical averaging
SENTIMENT_MAP = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}

# --- FinBERT Model Loading ---
print("Loading FinBERT model and tokenizer (ProsusAI/finbert)...")
try:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval() # Set model to evaluation mode
    print("FinBERT model loaded successfully.")
except Exception as e:
    print(f"Error loading FinBERT model: {e}")
    print("Please ensure 'transformers' and 'torch' libraries are installed.")
    print("Exiting as FinBERT is essential for sentiment analysis.")
    # In a notebook, `exit()` might stop the kernel. Consider raising an exception instead.
    raise SystemExit("FinBERT model loading failed.")

# --- Finnhub API Call Function ---
async def get_finnhub_company_news(session, ticker, start_date_str, end_date_str, api_key):
    """
    Fetches company news from Finnhub.io for a given ticker and date range.
    """
    url = f"{FINNHUB_BASE_URL}company-news?symbol={ticker}&from={start_date_str}&to={end_date_str}&token={api_key}"
    headers = {'Content-Type': 'application/json'}

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 429: # Rate limit exceeded
                print(f"Rate limit hit for {ticker} ({start_date_str} to {end_date_str}). Waiting before retrying...")
                await asyncio.sleep(60) # Wait for 60 seconds
                return await get_finnhub_company_news(session, ticker, start_date_str, end_date_str, api_key) # Retry
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            news_data = await response.json()
            return news_data
    except aiohttp.ClientError as e:
        print(f"HTTP error fetching news for {ticker} ({start_date_str} to {end_date_str}): {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred fetching news for {ticker} ({start_date_str} to {end_date_str}): {e}")
        return []

# --- Sentiment Analysis using FinBERT (Synchronous) ---
def get_sentiment_from_finbert(text):
    """
    Analyzes the sentiment of text using the loaded FinBERT model.
    Returns 'positive', 'negative', or 'neutral'.
    This function is synchronous and will be run in a thread pool by asyncio.to_thread.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

        # FinBERT's default id2label mapping for ProsusAI/finbert: 0: positive, 1: negative, 2: neutral
        if predicted_class_idx == 0:
            return 'positive'
        elif predicted_class_idx == 1:
            return 'negative'
        elif predicted_class_idx == 2:
            return 'neutral'
        else:
            print(f"Warning: Unexpected FinBERT output index {predicted_class_idx} for '{text}'. Defaulting to neutral.")
            return 'neutral'
    except Exception as e:
        print(f"Error during FinBERT sentiment analysis for '{text}': {e}. Defaulting to neutral.")
        return 'neutral'

# --- Main Data Collection and Processing Logic ---
async def collect_and_analyze_sentiment():
    all_data = []
    current_date = START_DATE

    # Calculate total weeks to process
    total_days = (END_DATE - START_DATE).days
    total_weeks = total_days // 7 + 1 # Add 1 for the last partial week

    # Use a session for all HTTP requests
    async with aiohttp.ClientSession() as session:
        week_count = 0
        while current_date <= END_DATE:
            week_count += 1
            week_end_date = current_date + timedelta(days=6)
            # Ensure week_end_date does not exceed END_DATE
            if week_end_date > END_DATE:
                week_end_date = END_DATE

            start_date_str = current_date.strftime('%Y-%m-%d')
            end_date_str = week_end_date.strftime('%Y-%m-%d')

            print(f"Processing week {week_count}/{total_weeks} ({start_date_str} to {end_date_str})")

            # Collect tasks for all tickers for the current week
            weekly_ticker_tasks = []
            for ticker in TICKERS:
                weekly_ticker_tasks.append(
                    get_finnhub_company_news(session, ticker, start_date_str, end_date_str, FINNHUB_API_KEY)
                )

            # Run all ticker news fetches concurrently for the week
            all_news_for_week = await asyncio.gather(*weekly_ticker_tasks)

            # Process news and sentiment for each ticker
            for i, ticker_news_list in enumerate(all_news_for_week):
                ticker = TICKERS[i]
                
                # Filter for relevant headlines (e.g., by taking the first few)
                # News articles from Finnhub have 'headline' and 'summary' fields
                headlines_for_sentiment = []
                for article in ticker_news_list:
                    if article.get('headline'): # Ensure headline exists
                        headlines_for_sentiment.append(article['headline'])
                    if len(headlines_for_sentiment) >= HEADLINES_PER_WEEK:
                        break # Stop after collecting enough headlines

                # If less than HEADLINES_PER_WEEK are found, pad with empty strings
                while len(headlines_for_sentiment) < HEADLINES_PER_WEEK:
                    headlines_for_sentiment.append("No relevant news found for this period.")

                # Run sentiment analysis on collected headlines concurrently
                sentiment_tasks = [asyncio.to_thread(get_sentiment_from_finbert, hl) for hl in headlines_for_sentiment]
                sentiments = await asyncio.gather(*sentiment_tasks)

                sentiment_scores = []
                headline_sentiments_dict = {}
                for j, sentiment_label in enumerate(sentiments):
                    numeric_sentiment = SENTIMENT_MAP.get(sentiment_label, 0)
                    sentiment_scores.append(numeric_sentiment)
                    headline_sentiments_dict[f'Headline_{j+1}'] = headlines_for_sentiment[j]
                    headline_sentiments_dict[f'Sentiment_{j+1}'] = sentiment_label

                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

                row_data = {
                    'Date_Week_Start': start_date_str,
                    'Date_Week_End': end_date_str,
                    'Ticker': ticker,
                    **headline_sentiments_dict,
                    'Average_Weekly_Sentiment': avg_sentiment
                }
                all_data.append(row_data)
            
            # --- IMPORTANT: Add a delay to avoid hitting Finnhub rate limits ---
            # Even with async, rapid consecutive calls can hit limits.
            # Adjust this sleep duration based on your Finnhub plan's rate limits.
            # Free tier is 30 calls/minute. 20 tickers/week means 20 calls.
            # If you process 1 week every 2 seconds, that's 10 calls/minute, which is safe.
            # For paid tiers, you might be able to reduce this.
            await asyncio.sleep(2) # Sleep for 2 seconds between processing each week's batch of tickers

            current_date += timedelta(days=7) # Move to the next week

    return pd.DataFrame(all_data)

# --- Execute the process and save to CSV ---
async def main():
    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        print("WARNING: Please replace 'YOUR_FINNHUB_API_KEY_HERE' with your actual Finnhub API key.")
        print("A paid Finnhub subscription is required for historical news data beyond 30 days.")
        return

    print("Starting sentiment analysis data collection...")
    df_sentiment = await collect_and_analyze_sentiment()

    print(f"\nData collection complete. Total rows: {len(df_sentiment)}")
    print("\nSample of collected data:")
    print(df_sentiment.head())

    # Save to CSV
    df_sentiment.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nData saved to {OUTPUT_CSV_FILE}")

    # Optional: Download the file in Colab
    # from google.colab import files
    # files.download(OUTPUT_CSV_FILE)

if __name__ == "__main__":
    # This block handles running the async main function directly in Colab
    # after nest_asyncio.apply() has been called.
    try:
        await main()
    except SystemExit as e:
        print(f"Execution stopped due to: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")


# --- Execute the process and save to CSV ---
async def main():
    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        print("WARNING: Please replace 'YOUR_FINNHUB_API_KEY_HERE' with your actual Finnhub API key.")
        print("A paid Finnhub subscription is required for historical news data beyond 30 days.")
        return

    print("Starting sentiment analysis data collection...")
    df_sentiment = await collect_and_analyze_sentiment()

    print(f"\nData collection complete. Total rows: {len(df_sentiment)}")
    print("\nSample of collected data:")
    print(df_sentiment.head())

    # Save to CSV
    df_sentiment.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nData saved to {OUTPUT_CSV_FILE}")

    # Optional: Download the file in Colab
    # from google.colab import files
    # files.download(OUTPUT_CSV_FILE)

if __name__ == "__main__":
    # This block handles running the async main function in different environments
    try:
        # Check if an event loop is already running (common in notebooks)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If a loop is running, schedule main as a task
            loop.create_task(main())
        else:
            # Otherwise, run the loop until main completes
            loop.run_until_complete(main())
    except RuntimeError as e:
        if "cannot run the event loop while another loop is running" in str(e):
            print("Detected existing event loop. Please restart your Colab runtime (Runtime -> Restart runtime)")
            print("and run the cell again. If the issue persists, try adding 'import nest_asyncio; nest_asyncio.apply()' at the very beginning of your script.")
        else:
            raise e
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")

