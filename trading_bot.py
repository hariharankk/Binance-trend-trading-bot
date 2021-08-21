# necessary libraries
import numpy as np
import pandas_ta as ta
from datetime import date
from datetime import timedelta
import pandas as pd
import json
import os
import time
import datetime
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
import configparser
import coinmarketcapapi
import smtplib
from binance.streams import BinanceSocketManager
import itertools
from threading import Thread  # defining client and apikeys


def client():
    api_key = ('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    api_secret = ('xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    # setting a client, with our keys
    client = Client(api_key, api_secret)
    return client


# json file stored in local, to read and write trades
def json1():
    coins_bought = {}
    coins_bought_file_path = 'coins_bought.json'
    if os.path.isfile(coins_bought_file_path):
        with open(coins_bought_file_path) as file:
            coins_bought = json.load(file)
    return coins_bought, coins_bought_file_path


# dynamically identify stable coins from the cmc api
def stablecoins(z):
    index = []
    for i in range(len(z)):
        if 'stablecoin' in z[i]:
            index.append(i)
    return index


# delete stablecoins
def delete_multiple_elements(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


# rank crypto based on last 7d ranking
def ranking(crypto, q):
    dummy = list(map(lambda coin: coin['USD'], q))
    dummy = list(map(lambda coin: coin['price'], dummy))
    crypto = [x for y, x in sorted(zip(dummy, crypto))]
    return crypto


# moving average
def ma(df, w):  # not meaningful name
    ma = df.rolling(window=w).mean()
    return ma


# find ma crossover
def crossover(d, d1):
    d2 = np.where(d1 > d, 1, 0)
    d3 = np.where(d > d1, -1, 0)
    d4 = d2 + d3
    return d4


def absdata(stablecoins, delete_multiple_elements, ranking):
    cmc = coinmarketcapapi.CoinMarketCapAPI(
        '12a455ad-70af-4051-8092-c6ac065c74b8')
    d = cmc.cryptocurrency_listings_latest() # not meaningful name
    crypto = list(map(lambda coin: coin['symbol'], d.data))
    tags = list(map(lambda coin: coin['tags'], d.data))
    index = stablecoins(tags)
    delete_multiple_elements(crypto, index)
    q = list(map(lambda coin: coin['quote'], d.data))
    delete_multiple_elements(q, index)
    Crypto = ranking(crypto[:10], q[:10])
    Crypto = list(map(lambda x: x + 'USDT', Crypto))
    return Crypto


# get current price
def get_price(coin):
    '''Return the current price for all coins on binance'''
    depth = client.get_symbol_ticker(symbol=coin)
    initial_price = float(depth['price'])
    return initial_price


# position sizing algorithm
def quantity():
    capital = client.get_asset_balance(asset='USDT')
    if float(capital['free']) > 0:
        # position sizing algorithms to size our capital, we don't trade more than 10% percent of our capital in a single trade
        pos = (float(capital['free']) * 0.1)
        if pos > 12:
            return pos
        else:
            return 0

"""
very important, binance needs corect step size as.
For example BTC supports a step size of 6 decimal points while XRP only supports 
one. So if we want to buy XRP, we need to make sure that the volume formatting is 
correct. Trying to buy 1.000 XRP would fail, while 1.0 would be executed.
"""

def convert_volume(coin):
    '''Converts the volume given in QUANTITY from USDT to the each coin's volume'''
    last_price = get_price(coin)
    QUANTITY = quantity()

    try:
        info = client.get_symbol_info(coin)
        step_size = info['filters'][2]['stepSize']
        lot_size = step_size.index('1') - 1

        if lot_size < 0:
            lot_size = 0

    except BinanceAPIException as e:
        print(e)

        # calculate the volume in coin from QUANTITY in USDT (default)
        volume = float(QUANTITY / float(last_price))

        # define the volume with the correct step size
        if lot_size < 0:
            volume = float('{:.1f}'.format(volume))

        else:
            volume = float('{:.{}f}'.format(volume, lot_size))

    return volume


# we update portfolio to update in json file
def update_porfolio(orders, trades):
    '''add every coin bought to our portfolio for tracking/selling later'''
    coinsbought, coins_bought_file_path = json1()
    for coin in orders:
        # add existing coin
        if coin in coinsbought:
            try:
                vol = float(coins_bought[coin]['volume'])
                coinsbought[coin] = {
                    'symbol': orders[coin][0]['symbol'],
                    'orderid': orders[coin][0]['orderId'],
                    'timestamp': orders[coin][0]['time'],
                    'bought_at': trades[coin][0]['price'],
                    'volume': float(trades[coin][0]['qty']) + vol
                }
            except TypeError as e:
                print(e)
        else:
            #add if coin in new
            try:
                coinsbought[coin] = {
                    'symbol': orders[coin][0]['symbol'],
                    'orderid': orders[coin][0]['orderId'],
                    'timestamp': orders[coin][0]['time'],
                    'bought_at': trades[coin][0]['price'],
                    'volume': float(trades[coin][0]['qty'])
                }
            except TypeError as e:
                print(e)

            # save the coins in a json file in the same directory
        with open(coins_bought_file_path, 'w') as file:
            json.dump(coinsbought, file, indent=4)


# generate buy signal
def buy(Crypto, y, t):

    for Crypto in Crypto: # not meaningful
        # data puller
        orders = {}
        trades = {}
        klines = client.get_historical_klines(Crypto,
                                              Client.KLINE_INTERVAL_1DAY, y, t)
        df = pd.DataFrame(klines,
                          columns=[
                              'timestamp', 'Open', 'High', 'Low', 'Close',
                              'volume', 'close_time', 'quote_av', 'trades',
                              'tb_base_av', 'tb_quote_av', 'ignore'
                          ])
        df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')

        #computing simple 50 day moving average
        df["50ma"] = ma(df['Close'], 50)
        #computing simple 100 day moving average
        df["100ma"] = ma(df['Close'], 100)
        # find moving average crossovers
        df['pos'] = crossover(df['100ma'], df['50ma'])

        # calculating upper bound or 50 day highest close
        df["upper_bound"] = df["High"].shift(1).rolling(window=50).max()
        # if today's close is higher than the prior 50 day high and we have golden crossover, we enter a trade
        if str(df['Close'].iloc[-1]) > str(df['upper_bound'].iloc[-1]) and str(
                df['pos'].iloc[-1]) == 1:

            smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
            smtp_server.ehlo()
            smtp_server.starttls()
            smtp_server.login('hkannan084@gmail.com', 'xxxxxxx')
            smtp_server.sendmail('hkannan084@gmail.com', 'xxxxxxxxxx', Crypto)
            smtp_server.quit()
            #get current price of the coin
            crypto = convert_volume(Crypto)
            if crypto != 0.0:
                try:
                    # don't execute this statement, enn order poirum :p, it works but (buy_order = client.create_order(symbol=ticker, side='BUY', type='MARKET', quantity=crypto))
                    while True:
                        orders[Crypto] = client.get_all_orders(symbol=Crypto,
                                                               limit=1)
                        trades[Crypto] = client.get_my_trades(symbol=Crypto,
                                                              limit=1)
                        if ((len(trades[Crypto]) > 0)
                                and (len(orders[Crypto]) > 0)):
                            break
                except BinanceAPIException as e:
                    # error handling goes here
                    print(e)
                except BinanceOrderException as e:
                    # error handling goes here
                    print(e)
            else:
                print("insufficient balance")

    update_porfolio(orders, trades)

# not so good comments/documentation
#sell block

"""
we need only 21 day prior data to calculate our indicators
we check this block only if we have trades
"""

def sell(coins_bought):
    stoploss = 0
    buy = {}
    if len(coins_bought) != 0:
        for buy in coins_bought:
            qty = coins_bought[buy]['volume']
            price = float(coins_bought[buy]['bought_at'])
            y1 = (datetime.datetime.fromtimestamp(
                coins_bought[buy]['timestamp'] / 1e3) -
                  timedelta(days=20)).strftime('%d %b, %Y')

            klines = client.get_historical_klines(buy,
                                                  Client.KLINE_INTERVAL_1DAY,
                                                  y1, t)
            df = pd.DataFrame(klines,
                              columns=[
                                  'timestamp', 'Open', 'High', 'Low', 'Close',
                                  'volume', 'close_time', 'quote_av', 'trades',
                                  'tb_base_av', 'tb_quote_av', 'ignore'
                              ])
            df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')

            # getting ticker
            df["lower_bound"] = df["Low"].shift(1).rolling(window=20).min()
            df['N'] = ta.atr(
                df.High.astype(float),
                df.Low.astype(float),
                df.Close.astype(float),
                length=20,
                mamode='sma',
                talib=None,
                drift=None,
                offset=None,
            )
            # find the open trades(price and qty)  that we have not exited using tickers
            # using average true range as trailing stoploss
            df['atr'] = float(price) - (3 * df['N'])
            stoploss = df['atr'].max()

            # we exit trade if our trailing stop loss is hit or our price closes below 20 day lowest
            if str(df['Close'].iloc[-1]) < str(
                    df['lower_bound'].iloc[-1]) or str(
                        df['Close'].iloc[-1]) < str(stoploss):
                try:

                    #don't execute this statement, enn order poirum :p, it works but (sell_order=client.create_order(symbol=coin, side='SELL', type='MARKET', quantity=qty))
                    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
                    smtp_server.ehlo()
                    smtp_server.starttls()
                    smtp_server.login('hkannan084@gmail.com', 'xxxxxxxxx')
                    smtp_server.sendmail('hkannan084@gmail.com', 'xxxxxx', buy)
                    smtp_server.quit()
                    coins_bought[Crypto] = None
                    with open(coins_bought_file_path, 'w') as file:
                        json.dump(coins_bought, file, indent=4)

                except BinanceAPIException as e:
                    print(e)


# find top 30 coin w.r.t to market cap and rank them in terms of 7day momentum, have to plug in your code here.
# test list
if __name__ == '__main__':
    start_time = time.time()
    #client=client()

    #load date to get data to compute indicators
    t = (date.today() - timedelta(days=1)).strftime('%d %b, %Y')
    y = (date.today() - timedelta(days=100)).strftime('%d %b, %Y')

    Crypto = absdata(stablecoins, delete_multiple_elements, ranking)
    # get trading universe w.r.t top 10coins by marketca[p] now
    coins_bought, coins_bought_file_path = json1()
    p1 = Thread(target=buy, args=(
        Crypto,
        y,
        t,
    ))
    p2 = Thread(target=sell, args=(coins_bought, ))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("MP--- %s seconds for single---" % (time.time() - start_time))
# since we need only 100 days data to compute our indicators, we find the dates to do so
