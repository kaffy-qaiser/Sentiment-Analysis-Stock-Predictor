import requests
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import praw
from dotenv import load_dotenv
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import dash
import dash_core_components as dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

# Load environment variables
load_dotenv()

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def fetch_reddit_data():
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
    subreddit = reddit.subreddit('wallstreetbets')
    posts = subreddit.new(limit=10000)
    data = []
    for post in posts:
        data.append([post.title, post.selftext, post.created_utc])
    return pd.DataFrame(data, columns=['title', 'body', 'created_utc'])

def preprocess_text(text):
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    if not isinstance(text, str):
        return 0
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def fetch_stock_data(ticker, start_date='2024-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.reset_index()
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    return stock_data

def prepare_data(merged_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_data[['sentiment', 'Close']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 1])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_model(X, y):
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 25
    for epoch in range(epochs):
        model.hidden_cell = (torch.zeros(1, X_train.size(0), model.hidden_layer_size),
                             torch.zeros(1, X_train.size(0), model.hidden_layer_size))
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    return model

def evaluate_model(model, X, scaler, merged_data):
    X_test = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, X_test.size(0), model.hidden_layer_size),
                             torch.zeros(1, X_test.size(0), model.hidden_layer_size))
        predicted_stock_price = model(X_test).numpy()
    dummy_array = np.zeros(predicted_stock_price.shape)
    predicted_stock_price_reshaped = np.concatenate((dummy_array, predicted_stock_price), axis=1)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price_reshaped)[:, 1]
    return merged_data['Date'], merged_data['Close'], predicted_stock_price

def fetch_and_process_data(ticker):
    reddit_data = fetch_reddit_data()
    reddit_data['cleaned_body'] = reddit_data['body'].apply(preprocess_text)
    reddit_data['sentiment'] = reddit_data['cleaned_body'].apply(get_sentiment)
    reddit_data['Date'] = pd.to_datetime(reddit_data['created_utc'], unit='s').dt.date

    reddit_data.to_csv('wsb_posts_sentiment.csv', index=False)

    stock_data = fetch_stock_data(ticker)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

    merged_data = pd.merge(reddit_data[['Date', 'sentiment']], stock_data, on='Date')

    X, y, scaler = prepare_data(merged_data)

    model = train_model(X, y)

    dates, actual_prices, predicted_prices = evaluate_model(model, X, scaler, merged_data)

    return dates, actual_prices, predicted_prices

def get_top_stocks():
    reddit_data = fetch_reddit_data()
    reddit_data['cleaned_body'] = reddit_data['body'].apply(preprocess_text)
    reddit_data['title'] = reddit_data['title'].apply(preprocess_text)

    all_text = ' '.join(reddit_data['title']) + ' ' + ' '.join(reddit_data['cleaned_body'])
    tokens = word_tokenize(all_text)
    tokens = [word for word in tokens if word.isalpha()]

    stock_counts = pd.Series(tokens).value_counts().head(5)
    return stock_counts.index.tolist()

def fetch_news(api_key, query='stock market', num_articles=100):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={num_articles}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles']
    return pd.DataFrame(articles)

def preprocess_and_analyze(articles_df):
    sia = SentimentIntensityAnalyzer()
    articles_df['sentiment'] = articles_df['description'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt']).dt.date
    return articles_df

def combine_data(news_df, stock_df):
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    combined_df = pd.merge(news_df, stock_df, left_on='publishedAt', right_on='Date')
    combined_df = combined_df[['Date', 'sentiment', 'Close', 'High', 'Low']]
    combined_df.drop_duplicates(subset=['Date'], inplace=True)
    return combined_df

def calculate_stochastic_oscillator(df, window=14, smooth_window=3):
    df['Lowest_Low'] = df['Low'].rolling(window=window).min()
    df['Highest_High'] = df['High'].rolling(window=window).max()
    df['%K'] = ((df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
    df['%D'] = df['%K'].rolling(window=smooth_window).mean()
    return df

def train_and_evaluate(df, sentiment_col='sentiment'):
    X = df[[sentiment_col]]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    latest_k = df['%K'].iloc[-1]
    latest_d = df['%D'].iloc[-1]

    recommendations = []
    for prediction, actual in zip(predictions, y_test):
        if latest_k < 20 and latest_k > latest_d:
            recommendations.append('Buy')
        elif latest_k > 80 and latest_k < latest_d:
            recommendations.append('Sell')
        else:
            recommendations.append('Hold')

    unique_dates = set()
    results = []
    for idx, (prediction, actual, recommendation) in enumerate(zip(predictions, y_test, recommendations)):
        date = X_test.index[idx]
        if df.loc[date, 'Date'] not in unique_dates:
            result = {
                "Date": df.loc[date, 'Date'],
                "Predicted Price": prediction,
                "Actual Price": actual,
                "Recommendation": recommendation
            }
            results.append(result)
            unique_dates.add(df.loc[date, 'Date'])

    return model, results, mse

# Dash app setup
app = dash.Dash(__name__)
app.title = "Stock Price Prediction Dashboard"
app.layout = html.Div(
    style={'fontFamily': 'Arial', 'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1("Sentiment Analysis on Stocks", style={'textAlign': 'center', 'color': '#333'}),
        html.Div([
            dcc.Input(id='ticker-input', type='text', placeholder='Enter stock ticker', value='AAPL',
                      style={'width': '200px', 'padding': '10px', 'marginRight': '10px'}),
            html.Button('Submit', id='submit-button', n_clicks=0,
                        style={'padding': '10px', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=[dcc.Graph(id='stock-graph')],
            style={'display': 'block', 'margin': 'auto'}
        ),
        html.H2("Top 5 Most Used Words on r/wallstreetbets", style={'textAlign': 'center', 'color': '#333'}),
        html.Div(id='top-stocks', style={'textAlign': 'center', 'marginTop': '20px'}),
        dcc.Loading(
            id="loading-2",
            type="default",
            children=[
                dcc.Graph(id='news-sentiment-graph'),
                dcc.Graph(id='combined-sentiment-graph')
            ],
            style={'display': 'block', 'margin': 'auto'}
        )
    ]
)

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def update_graph(n_clicks, ticker):
    if n_clicks > 0:
        dates, actual_prices, predicted_prices = fetch_and_process_data(ticker)
        figure = {
            'data': [
                go.Scatter(x=dates, y=actual_prices, mode='lines', name='Actual Price'),
                go.Scatter(x=dates[60:], y=predicted_prices, mode='lines', name='Predicted Price')
            ],
            'layout': {
                'title': f'Reddit Sentiment Analysis for {ticker}',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Stock Price'},
                'plot_bgcolor': '#f9f9f9',
                'paper_bgcolor': '#f9f9f9'
            }
        }
        return figure
    return {}

@app.callback(
    Output('top-stocks', 'children'),
    [Input('submit-button', 'n_clicks')]
)
def update_top_stocks(n_clicks):
    if n_clicks > 0:
        top_stocks = get_top_stocks()
        return html.Ul([html.Li(stock) for stock in top_stocks])
    return html.Ul([html.Li("Enter a stock ticker")])

@app.callback(
    [Output('news-sentiment-graph', 'figure'),
     Output('combined-sentiment-graph', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def update_additional_graphs(n_clicks, ticker):
    if n_clicks > 0:
        api_key = os.getenv('NEWS_API_KEY')
        news_df = fetch_news(api_key)
        news_df = preprocess_and_analyze(news_df)

        today = datetime.today().strftime('%Y-%m-%d')
        stock_df = fetch_stock_data(ticker, '2024-01-01', today)
        combined_df = combine_data(news_df, stock_df)

        combined_df = calculate_stochastic_oscillator(combined_df)

        print("Evaluating model with news sentiment data:")
        news_model, news_results, news_mse = train_and_evaluate(combined_df)

        wsb_df = pd.read_csv('wsb_posts_sentiment.csv')
        wsb_df['Date'] = pd.to_datetime(wsb_df['created_utc'], unit='s').dt.date
        wsb_df['Date'] = pd.to_datetime(wsb_df['Date'])

        # Ensure both Date columns are of the same type
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        wsb_df['Date'] = pd.to_datetime(wsb_df['Date'])

        combined_df = pd.merge(combined_df, wsb_df, on='Date', how='inner', suffixes=('_news', '_wsb'))
        combined_df['combined_sentiment'] = combined_df[['sentiment_news', 'sentiment_wsb']].mean(axis=1)

        print("Evaluating combined model with news and r/wallstreetbets sentiment data:")
        combined_model, combined_results, combined_mse = train_and_evaluate(combined_df, sentiment_col='combined_sentiment')

        news_results = sorted(news_results, key=lambda x: x['Date'])
        combined_results = sorted(combined_results, key=lambda x: x['Date'])

        news_data = pd.DataFrame(news_results)
        combined_data = pd.DataFrame(combined_results)

        fig1 = {
            'data': [
                {
                    'x': news_data['Date'],
                    'y': news_data['Predicted Price'],
                    'type': 'line',
                    'name': 'Predicted Price (News Sentiment)',
                    'customdata': news_data['Recommendation'],
                    'hovertemplate': 'Date: %{x}<br>Predicted Price: %{y}<br>Recommendation: %{customdata}<extra></extra>'
                },
                {
                    'x': news_data['Date'],
                    'y': news_data['Actual Price'],
                    'type': 'line',
                    'name': 'Actual Price (News Sentiment)',
                    'customdata': news_data['Recommendation'],
                    'hovertemplate': 'Date: %{x}<br>Actual Price: %{y}<br>Recommendation: %{customdata}<extra></extra>'
                }
            ],
            'layout': {
                'title': f'Stock Prices Predicted vs Actual (News Sentiment) - MSE: {news_mse:.2f}'
            }
        }

        fig2 = {
            'data': [
                {
                    'x': combined_data['Date'],
                    'y': combined_data['Predicted Price'],
                    'type': 'line',
                    'name': 'Predicted Price (News & WSB Sentiment)',
                    'customdata': combined_data['Recommendation'],
                    'hovertemplate': 'Date: %{x}<br>Predicted Price: %{y}<br>Recommendation: %{customdata}<extra></extra>'
                },
                {
                    'x': combined_data['Date'],
                    'y': combined_data['Actual Price'],
                    'type': 'line',
                    'name': 'Actual Price (News & WSB Sentiment)',
                    'customdata': combined_data['Recommendation'],
                    'hovertemplate': 'Date: %{x}<br>Actual Price: %{y}<br>Recommendation: %{customdata}<extra></extra>'
                }
            ],
            'layout': {
                'title': f'Stock Prices Predicted vs Actual (News & WSB Sentiment) - MSE: {combined_mse:.2f}'
            }
        }

        return fig1, fig2

    return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)


