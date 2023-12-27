from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import json
import plotly
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime


app = Flask(__name__)

# Load the coin data from the main CSV file
data = pd.read_csv("crypto-markets copy.csv")
coin_symbols = {'bitcoin': 'BTC', 'ethereum': 'ETH', 'litecoin': 'LTC'}

def get_fig(selected_coin):
    df = data[data['symbol'] == coin_symbols[selected_coin]]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    fig = px.line(df, x=df.index, y='close', title=f'{selected_coin} - USD Time Series')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
def train_and_predict(df, target_date):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['close'].values
    model = ExtraTreesRegressor(n_estimators=500, min_samples_split=5)
    model.fit(X, y)

    days_since_start = (target_date - df.index.min()).days
    predicted_price = model.predict(np.array(days_since_start).reshape(-1, 1))[0]

    return predicted_price
    

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_coin = 'bitcoin'
    investment_date = ''
    investment_amount = 0
    result_message = ''
    investment_value = 0
    target_date = ''
    target_value = 0

    if request.method == 'POST':
        selected_coin = request.form['coin']
        investment_date = request.form['investment_date']
        investment_amount = float(request.form['investment_amount'])
        target_date = request.form['target_date']

        df = data[data['symbol'] == coin_symbols[selected_coin]]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        investment_date = pd.Timestamp(investment_date)
        target_date = pd.Timestamp(target_date)

        # Predict the price on the investment_date
        predicted_price = train_and_predict(df, investment_date)

        # Predict the price on the target_date
        predicted_price_on_target_date = train_and_predict(df, target_date)

        investment_value = investment_amount
        target_value = predicted_price_on_target_date
        profit_loss = target_value - investment_value

        if profit_loss >= 0:
            result_message = f"Potential Profit: ${profit_loss:.2f}"
        else:
            result_message = f"Potential Loss: ${-profit_loss:.2f}"

    graphJSON = get_fig(selected_coin)

    return render_template('crypto_tracker.html',
                           coins=coin_symbols.keys(),
                           selected_coin=selected_coin,
                           graphJSON=graphJSON,
                           investment_date=investment_date,
                           investment_amount=investment_amount,
                           target_date=target_date,
                           predicted_value=target_value,
                           result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)