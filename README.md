# Sentiment Analysis Stock Dashboard

## Overview

This project is a dashboard that utilizes sentiment analysis from Reddit's r/wallstreetbets and news articles to predict stock prices. It leverages natural language processing (NLP), machine learning (ML), and deep learning (DL) techniques to analyze sentiment and predict stock price trends. The dashboard is built using Python with Dash for the web interface.

## Features

- **Sentiment Analysis:** Analyzes sentiment from Reddit posts and news articles.
- **Stock Price Prediction:** Uses LSTM (Long Short-Term Memory) neural network to predict stock prices based on sentiment data.
- **Visualization:** Displays actual vs. predicted stock prices and sentiment trends using Plotly graphs.

## Technologies Used

- **Python Libraries:**
  - `requests`
  - `yfinance`
  - `pandas`
  - `numpy`
  - `torch`
  - `nltk`
  - `scikit-learn`
  - `vaderSentiment`
  - `dash`
  - `plotly`
- **Reddit API:** PRAW (Python Reddit API Wrapper)
- **News API:** For fetching news articles
- **Yahoo Finance API:** For fetching stock data

## Prerequisites

- Python 3.8+
- Anaconda or virtual environment
- Reddit API credentials
- News API key

## Setup and Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/kaffy-qaiser/stock-price-prediction-dashboard.git
    cd stock-price-prediction-dashboard
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory of the project.
    - Add your Reddit API credentials and News API key to the `.env` file:
        ```env
        REDDIT_CLIENT_ID=your_reddit_client_id
        REDDIT_CLIENT_SECRET=your_reddit_client_secret
        REDDIT_USER_AGENT=your_reddit_user_agent
        NEWS_API_KEY=your_news_api_key
        ```

5. **Download NLTK resources:**
    ```sh
    python -m nltk.downloader stopwords
    python -m nltk.downloader punkt
    python -m nltk.downloader vader_lexicon
    ```

## Usage

1. **Run the Dash app:**
    ```sh
    python app.py
    ```

2. **Open the dashboard:**
    - Navigate to `http://127.0.0.1:8050/` in your web browser.

3. **Interact with the dashboard:**
    - Enter a stock ticker (e.g., AAPL) and click the "Submit" button.
    - View the actual vs. predicted stock prices and sentiment trends.


## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

- The NLTK and VADER Sentiment libraries for natural language processing and sentiment analysis.
- The Dash and Plotly libraries for building interactive web applications and visualizations.
- Yahoo Finance and News API for providing financial data and news articles.
- Reddit and the r/wallstreetbets community for the data used in this project.

---

