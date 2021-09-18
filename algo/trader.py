# necessary libraries
import numpy as np
import pandas_ta as ta
from datetime import date
from datetime import timedelta
import math
import pandas as pd
import json
import ctypes, sys
import time
import datetime
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.client import Client
import import_ipynb
import mail
import testapi
import visuvalization
import portfolio
from threading import Thread
import Cred

# moving average 
def stablecoins(z):
    index=[]
    for i in range(len(z)):
        if 'stablecoin' in z[i] or 'wrapped-tokens' in z[i]:
            index.append(i)
    return index

# delete stablecoins 
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def ranking(crypto,q):
    dummy=list(map(lambda coin: coin['USD'], q))
    dummy=list(map(lambda coin: coin['price'], dummy)) 
    crypto=[x for y, x in sorted(zip( dummy, crypto))]
    return crypto

def absdata():    
    cmc = Cred.cmc_main()    
    data = cmc.cryptocurrency_listings_latest()
    crypto= list(map(lambda coin: coin['symbol'], data.data))
    tags=list(map(lambda coin: coin['tags'], data.data))
    index=stablecoins(tags)                
    delete_multiple_element(crypto, index)            
    q=list(map(lambda coin: coin['quote'], data.data))
    delete_multiple_element(q, index)
    Crypto=ranking(crypto[:20],q[:20])
    Crypto = list(map(lambda x: x + 'USDT'  , Crypto))
    return Crypto

def ma(df,w):
    ma=ta.tema(df, length=w, talib=None, offset=None)
    return ma

def sell_stoploss(df):
    df['N']=atr(df)
    # find the open trades(price and qty)  that we have not exited using tickers
    # using average true range as trailing stoploss
    df['atr']= df.Close.astype(float) - ( 3 * df['N']) 
    for i in range(1,len(df['atr'])):
        if df['atr'].iloc[i]<df['atr'].iloc[i-1]:
            df.loc[i,'atr']=df['atr'].iloc[i-1]
    stoploss = df['atr'][:len(df['atr'])-1].max()
    return stoploss
def atr(df):
  atr=ta.atr(df.High.astype(float), df.Low.astype(float), df.Close.astype(float), length=20, mamode='sma', talib=None, drift=None, offset=None,)
  return atr

def sell_ma(Crypto,w1,w2):
  t = (date.today() - timedelta(days = 1)).strftime('%d %b, %Y')
  y = (date.today() - timedelta(days = 100)).strftime('%d %b, %Y')
  
  try:
    klines = client.get_historical_klines(Crypto, Client.KLINE_INTERVAL_1DAY, y, t)
    df = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
  except BinanceAPIException as e:
    msg='data retrieval failed'
    mail.send_mail(msg)

  avg1=ma(pd.to_numeric(df['Close']),w1)
  avg2=ma(pd.to_numeric(df['Close']),w2)
  if avg1.iloc[-1] > avg2.iloc[-1]:
    return 1
  else:
      return 0
# find ma crossover
def crossover(d,d1):
    d2=np.where(d1>d,1,0)
    d3=np.where(d>d1,-1,0)
    d4=d2+d3
    return d4

def get_order_status(currentOrder):
  order =  currentOrder['status'] 
  if  order == 'FILLED':
    return order
  else: return 'EMPTY'  

# get current price
def get_price(coin):
    '''Return the current price for all coins on binance'''
    try:
      depth = client.get_symbol_ticker(symbol=coin)
    except BinanceAPIException as e:
      msg='get price failed'
      mail.send_mail(msg)
    
    initial_price=float(depth['price'])
    return initial_price        

# position sizing algorithm
def quantity(coin,df):
    try: 
      capital=client.get_asset_balance(asset='USDT')
    except BinanceAPIException as e:
      msg='failed while getting usdt balance'
      mail.send_mail(msg)
    last_price=get_price(coin)
    
    if float(capital['free']) > 0:
        # position sizing algorithms to size our capital, we don't trade more than 10% percent of our capital in a single trade
        dummy=atr(df)
        dummy=dummy.iloc[-1]
        pos = float(float(capital['free']) * 0.02/dummy)
        p_size= pos * float(last_price)
        print(p_size)
        if p_size > 12 and p_size < float(capital['free']):
            return pos
        else:
            return 0

def min_sell(coins_bought,buy):
  qty=coins_bought[buy]['volume']     
  price=float(coins_bought[buy]['bought_at'])
  price=get_price(buy)
  if price*qty > 12:
    return 1
  else:
    return 0   

#very important, binance needs corect step size as For example BTC supports a step size of 6 decimal points while XRP only supports one. So if we want to buy XRP, we need to make sure that the volume formatting is correct. Trying to buy 1.000 XRP would fail, while 1.0 would be executed.
def convert_volume(coin,df):
    '''Converts the volume given in QUANTITY from USDT to the each coin's volume'''
    QUANTITY=quantity(coin,df)
    try:
        info = client.get_symbol_info(coin)
        step_size = info['filters'][2]['stepSize']
        lot_size = step_size.index('1') - 1

        if lot_size < 0:
            lot_size = 0
    
    except BinanceAPIException as e:
        msg='error geting volume stepsize'
        mail.send_mail(msg)
        # calculate the volume in coin from QUANTITY in USDT (default)
    volume = float(QUANTITY)
        # define the volume with the correct step size
    if  lot_size < 0:
        volume = float('{:.1f}'.format(volume))
    else:
        volume = float('{:.{}f}'.format(volume, lot_size))
    return volume

# we update portfolio to update in json file
# generate buy signal

def buy(crypto_set,y,t,coins_bought):
    orders={}
    trades={}
    for Crypto in crypto_set:
            # data puller 
            try:
              klines = client.get_historical_klines(Crypto, Client.KLINE_INTERVAL_1DAY, y, t)
              df = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
              df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
           
              if Crypto in coins_bought:
                  count=coins_bought[Crypto]['count']            
              else:
                  count=0
            #computing simple 50 day moving average
            
              df["50ma"] = ma(pd.to_numeric(df['Close']),50)
            #computing simple 100 day moving average
              df["100ma"] = ma(pd.to_numeric(df['Close']),100)
            # find moving average crossovers
              df['pos']=crossover(df['100ma'],df['50ma'])
            # calculating upper bound or 50 day highest close
              df["upper_bound"] = df["High"].shift(1).rolling(window=50).max()
            # if today's close is higher than the prior 50 day high and we have golden crossover, we enter a trade
              
              if float(df['Close'].iloc[-1]) > float(df['upper_bound'].iloc[-1]) and float(df['pos'].iloc[-1]) == 1 and count < 2 :
                #get current price of the coin
                  position=convert_volume(Crypto,df)
                  if position != 0.0:
                      try:
                       
                        buy_order = client.create_order(symbol=Crypto, side='BUY', type='MARKET', quantity=position)
                        status=get_order_status(buy_order)
                        msg='we have bought '+str(Crypto)+'. the order status is '+str(status)+'. the quantity is '+str(position)
                        mail.send_mail(msg)
                        while True:   
                          orders[Crypto] = client.get_all_orders(symbol=Crypto, limit=1)    
                          trades[Crypto] = client.get_my_trades(symbol=Crypto, limit=1)   
                          if((len(trades[Crypto]) > 0) and (len(orders[Crypto]) > 0)):
                             break                       
                      except BinanceAPIException as e:
                          msg="Api execution failed buy "
                            # error handling goes here
                          mail.send_mail(msg)
                      except BinanceOrderException as e:                       
                          msg="order execution failed"  # error handling goes here
                          mail.send_mail(msg)
                  else:
                      msg="insufficient balance"
                      mail.send_mail(msg)
            except BinanceAPIException as e:
              msg='data retrieval failed'
              mail.send_mail(msg)
    
    portfolio.update_porfolio(orders,trades)        
#sell block
# we need only 21 day prior data to calculate our indicators
# we check this block only if we have trades
def sell(coins_bought):
    stoploss=0             
    buy={}
    if len(coins_bought) != 0:
        for buy in coins_bought.copy():
            qty=coins_bought[buy]['volume']            
            price=float(coins_bought[buy]['bought_at'])
            y1=(datetime.datetime.fromtimestamp(coins_bought[buy]['timestamp'] / 1e3)- timedelta(days = 22)).strftime('%d %b, %Y')
            check=min_sell(coins_bought,buy)
            if check == 1:
              try:
                klines = client.get_historical_klines(buy, Client.KLINE_INTERVAL_1DAY, y1, t)
                df = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
                df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
              
              # getting ticker
                df["lower_bound"] = df["Low"].shift(1).rolling(window=20).min()
                stoploss=sell_stoploss(df)
                movingavg=sell_ma(buy,50,100)  
                #msg='we have bought '+str(buy)+'. the stoploss is '+str(stoploss)+'. the closeprice is '+str(df['Close'].iloc[-1])+'. the lowerbound is '+str(df['lower_bound'].iloc[-1])
                #print(msg)
                visuvalization.vis(df,buy)

                          # we exit trade if our trailing stop loss is hit or our price closes below 20 day lowest
                if float(df['Close'].iloc[-1]) < float(df['lower_bound'].iloc[-1]) or float(df['Close'].iloc[-1]) < float(stoploss) or movingavg == 0:
                    try:
                        
                        sell_order = client.order_market_sell(symbol=buy, quantity=qty)
                        status=get_order_status(sell_order)
                        msg='we have bought '+str(buy)+'. the order status is '+str(status)+'. the quantity is '+str(qty)
                        mail.send_mail(msg)
                        del coins_bought[buy] 
                        with open(coins_bought_file_path, 'w') as file:
                            json.dump(coins_bought, file, indent=4)
                  
                    except BinanceAPIException as e:
                        msg="api execution failed sell"
                        mail.send_mail(msg)
                    except BinanceOrderException as e:    
                        msg="order execution failed"  # error handling goes here
                        mail.send_mail(msg)    
              except BinanceAPIException as e:
                msg="data retrival failed"
                mail.send_mail(msg)
            else:
              msg='insufficient coins to sell'
              mail.send_mail(msg)

# find top 30 coin w.r.t to market cap and rank them in terms of 7day momentum, have to plug in your code here.
# test list
if __name__ == '__main__':
    start_time = time.time()

    global client    
    client=Cred.main()
    api_ready, msg = testapi.test_api_key(client, BinanceAPIException)
    msg1=testapi.server_status(client)
    msg=str(msg)+'  '+str(msg1)
    mail.send_mail(msg)
    #load date to get data to compute indicators
    t=(date.today() - timedelta(days = 1)).strftime('%d %b, %Y')
    y = (date.today() - timedelta(days = 100)).strftime('%d %b, %Y')

    Crypto=absdata()
    # get trading universe w.r.t top 10coins by marketcap now    
    coins_bought,coins_bought_file_path=portfolio.json1()    
    p1 = Thread(target=buy,args=(Crypto,y,t,coins_bought))
    p2 = Thread(target=sell,args=(coins_bought,))    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("MP--- %s seconds for single---" % (time.time() - start_time))
# since we need only 100 days data to compute our indicators, we find the dates to do so
    
