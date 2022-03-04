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
import PNLSTATEMENT as p
import joblib


class cryptocmcdata():
   def __init__(self):
     pass


   def stablecoins(self,tags):
      index=[]
      for i in range(len(tags)):
          if 'stablecoin' in tags[i] or 'wrapped-tokens' in tags[i]:
              index.append(i)
      return index

  # delete stablecoins 
   def delete_multiple_element(self,list_object, indices):
      indices = sorted(indices, reverse=True)
      for idx in indices:
          if idx < len(list_object):
              list_object.pop(idx)

   def ranking(self,crypto,q):
      dummy=list(map(lambda coin: coin['USD'], q))
      dummy=list(map(lambda coin: coin['price'], dummy)) 
      crypto=[x for y, x in sorted(zip( dummy, crypto))]
      return crypto

   def absdata(self):    
      cmc = Cred.cmc_main()    
      data = cmc.cryptocurrency_listings_latest()
      crypto= list(map(lambda coin: coin['symbol'], data.data))
      tags=list(map(lambda coin: coin['tags'], data.data))
      index=self.stablecoins(tags)                
      self.delete_multiple_element(crypto, index)            
      q=list(map(lambda coin: coin['quote'], data.data))
      self.delete_multiple_element(q, index)
      Crypto=self.ranking(crypto[:20],q[:20])
      Crypto = list(map(lambda x: x + 'USDT'  , Crypto))
      return Crypto

# moving average 
class trader(cryptocmcdata):
  def __init__(self):
      self.client=Cred.main()
      self.y=(date.today() - timedelta(days = 200)).strftime('%d %b, %Y')
      self.t=(date.today() - timedelta(days = 1)).strftime('%d %b, %Y')
      self.df_btc = self.data('BTCUSDT',self.y)
      self.coins_bought, self.coins_bought_file_path=portfolio.json1() 
      self.crypto_set= cryptocmcdata.absdata(self)

  def get_order_status(self,currentOrder):
    order =  currentOrder['status'] 
    if  order == 'FILLED':
      return order
    else: return 'EMPTY'  

# get current price
  def get_price(self,coin):
      '''Return the current price for all coins on binance'''
      try:
        depth = self.client.get_symbol_ticker(symbol=coin)
      except BinanceAPIException as e:
        msg='get price failed'
        mail.send_mail(msg)     
      initial_price=float(depth['price'])
      return initial_price        


  def atr(self,df):
    atr=ta.atr(df.High.astype(float), df.Low.astype(float), df.Close.astype(float), length=20, mamode='sma', talib=None, drift=None, offset=None,)
    return atr

  def sell_stoploss(self,df):
      df['N']=self.atr(df)
      # find the open trades(price and qty)  that we have not exited using tickers
      # using average true range as trailing stoploss
      df['atr']= df.Close.astype(float) - ( 3 * df['N']) 
      for i in range(1,len(df['atr'])):
          if df['atr'].iloc[i]<df['atr'].iloc[i-1]:
              df.loc[i,'atr']=df['atr'].iloc[i-1]
      stoploss = df['atr'][:len(df['atr'])-1].max()
      return stoploss


  def sell_ma(self,Crypto):    
    try:
      df=self.data(Crypto,self.y)
    except BinanceAPIException as e:
      msg='indicator data retrieval failed'
      mail.send_mail(msg)
      10,25
    avg1=ta.ema(df['Close'].astype(float), length=10)
    avg2=ta.ema(df['Close'].astype(float), length=25)
    if avg1.iloc[-1] > avg2.iloc[-1]:
      return 1
    else:
      return 0

  def min_sell(self, buy):
    qty=float(self.coins_bought[buy]['volume'])     
    price=self.get_price(buy)
    if price*qty > 10:
      return 1
    else:
      return 0   

# find ma crossover
  def crossover(self,d,d1):
      d2=np.where(d1>d,1,0)
      d3=np.where(d>d1,-1,0)
      d4=d2+d3
      return d4


# position sizing algorithm
  def quantity(self,coin,df):
      try: 
        capital=self.client.get_asset_balance(asset='USDT')
      except BinanceAPIException as e:
        msg='failed while getting usdt balance'
        mail.send_mail(msg)
      last_price=self.get_price(coin)
      
      if float(capital['free']) > 0:
          # position sizing algorithms to size our capital, we don't trade more than 10% percent of our capital in a single trade
          dummy=self.atr(df)
          dummy=dummy.iloc[-1]
          pos = float(float(capital['free']) * 0.02/dummy)
          p_size= pos * float(last_price)
          print(p_size)
          if p_size > 12 and p_size < float(capital['free']):
              return pos
          else:
              return 0


#very important, binance needs corect step size as For example BTC supports a step size of 6 decimal points while XRP only supports one. So if we want to buy XRP, we need to make sure that the volume formatting is correct. Trying to buy 1.000 XRP would fail, while 1.0 would be executed.
  def convert_volume(self,coin,df):
      '''Converts the volume given in QUANTITY from USDT to the each coin's volume'''
      QUANTITY=self.quantity(coin,df)
      try:
          info = self.client.get_symbol_info(coin)
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
  
  def indicator(self,df_1d):
      df_1d['Close']=df_1d.Close.astype(float)
      df_1d['High']=df_1d.High.astype(float)
      df_1d['Low']=df_1d.Low.astype(float)
      df_1d['Open']=df_1d.Open.astype(float)
      df_1d['Close']=df_1d.Close.astype(float)
      df_1d['Volume']=df_1d.volume.astype(float)
      df_1d['atr']=ta.atr(df_1d.High, df_1d.Low, df_1d.Close, length=20, mamode='sma', talib=None, drift=None, offset=None, )
      df_1d['rsi']=ta.rsi(df_1d['Close'], length=14, scalar=100, drift=1)
      df_1d['200ma']=ta.sma(df_1d['Close'], length=200)
      df_1d['obv']=ta.obv(df_1d['Close'], df_1d['volume'].astype(float))
      df_1d['returns'] = np.log(df_1d['Close'] / df_1d['Close'].shift(1))
      df_1d['Volatility'] = df_1d['returns'].rolling(window=50).std() * np.sqrt(50)
      df_1d["100ma"] = ta.ema(df_1d['Close'], length=25)
      df_1d["50ma"] = ta.ema(df_1d['Close'], length=10)
      df_1d["upperbound"] = df_1d["High"].rolling(window=20).max()
      df_1d["lowerbound"] = df_1d["Low"].rolling(window=10).max()
      df_1d['willr']=ta.willr(df_1d['High'], df_1d['Low'], df_1d['Close'], length=14)
      df_1d['chop']=ta.chop(df_1d['High'], df_1d['Low'], df_1d['Close'])
      return df_1d

  def data(self,buy,y):                
      klines = self.client.get_historical_klines(buy, Client.KLINE_INTERVAL_1DAY, y, self.t)
      df = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
      df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
      return df
  
  def beta_data(self, df_1d):
      self.df_btc['ret']=self.df_btc['Close'].astype(float).pct_change()
      if float(self.df_btc['ret'].iloc[np.where(df_1d['timestamp'].iloc[-1]==self.df_btc['timestamp'])]) > 0:
        beta=1
      else:
        beta=0
      return beta        


  def buy(self):
      orders={}; trades={}
      for Crypto in self.crypto_set:
              # data puller 
              try:
                df=self.data(Crypto,self.y)
                if Crypto in self.coins_bought:
                    count=self.coins_bought[Crypto]['count']            
                else:
                    count=0
              #computing simple 50 day moving average
                df["50ma"] = ta.ema(df['Close'].astype(float), length=10)
              #computing simple 100 day moving average
                df["100ma"] = ta.ema(df['Close'].astype(float), length=25)
              # find moving average crossovers
                df['pos']=self.crossover(df['100ma'],df['50ma'])
              # calculating upper bound or 50 day highest close
                df["upper_bound"] = df["High"].shift(1).rolling(window=20).max()
              # if today's close is higher than the prior 50 day high and we have golden crossover, we enter a trade
                
                if float(df['Close'].iloc[-1]) > float(df['upper_bound'].iloc[-1]) and float(df['pos'].iloc[-1]) == 1 and count < 2 :
                  #get current price of the coin
                    position=self.convert_volume(Crypto,df)
                    if position != 0.0:
                        df=df.drop(['50ma', 'pos', '100ma', 'upper_bound' , 'close_time', 'quote_av', 'trades','tb_base_av','tb_quote_av','ignore'],axis=1)
                        df=self.indicator(df)
                        df=df.drop(['returns','volume'],axis=1)
                        beta=self.beta_data(df)
                        df=df.drop(['timestamp','Open','Low'],axis=1)
                        df=df.round(5)
                        rf = joblib.load('random_forest.joblib')
                        pred=int(rf.predict(df.tail(1)))
                        if(pred==1):
                          try:
                            buy_order = self.client.create_order(symbol=Crypto, side='BUY', type='MARKET', quantity=position)
                            status=self.get_order_status(buy_order)
                            msg='we have bought '+str(Crypto)+'. the order status is '+str(status)+'. the quantity is '+str(position)
                            mail.send_mail(msg)
                            while True:   
                              orders[Crypto] = self.client.get_all_orders(symbol=Crypto, limit=1)    
                              trades[Crypto] = self.client.get_my_trades(symbol=Crypto, limit=1)   
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
                          msg="Artificial Intelligence told us to not take a trade :p"
                          mail.send_mail(msg)

                    else:
                        msg="insufficient balance"
                        mail.send_mail(msg)
              except BinanceAPIException as e:
                msg='buy side data retrieval failed'
                mail.send_mail(msg)
      
      portfolio.update_porfolio(orders,trades)        
#sell block
# we need only 21 day prior data to calculate our indicators
# we check this block only if we have trades
  def sell(self):
      stoploss=0; buy={}
      if len(self.coins_bought) != 0:
          for buy in self.coins_bought.copy():          
              qty=float(self.coins_bought[buy]['volume'])            
              price=float(self.coins_bought[buy]['bought_at'])
              y=(datetime.datetime.fromtimestamp(self.coins_bought[buy]['timestamp'] / 1e3)- timedelta(days = 22)).strftime('%d %b, %Y')
              check=self.min_sell(buy)
              if check == 1:
                try:
                # getting ticker
                  df=self.data(buy,y)
                  df["lower_bound"] = df["Low"].shift(1).rolling(window=10).min()
                  stoploss=self.sell_stoploss(df)
                  movingavg=self.sell_ma(buy)  
                  visuvalization.vis(df,buy)
                  # we exit trade if our trailing stop loss is hit or our price closes below 20 day lowest
                  if float(df['Close'].iloc[-1]) < float(df['lower_bound'].iloc[-1]) or float(df['Close'].iloc[-1]) < float(stoploss) or movingavg == 0:
                      try:
                          sell_order = self.client.order_market_sell(symbol=buy, quantity=qty)
                          #sell_order=self.client.create_order(symbol=buy,side='SELL',type=Client.ORDER_TYPE_MARKET,quantity=qty)
                          status=self.get_order_status(sell_order)
                          msg='we have sold '+str(buy)+'. the order status is '+str(status)+'. the quantity is '+str(qty)
                          mail.send_mail(msg)
                          del self.coins_bought[buy] 
                          with open(self.coins_bought_file_path, 'w') as file:
                              json.dump(self.coins_bought, file, indent=4)
                      except BinanceAPIException as e:
                          msg="api execution failed sell"
                          mail.send_mail(msg)
                      except BinanceOrderException as e:    
                          msg="order execution failed"  # error handling goes here
                          mail.send_mail(msg)    
                except BinanceAPIException as e:
                  msg="sell side data retrival failed"
                  mail.send_mail(msg)
              else:
                msg='insufficient coins to sell'
                mail.send_mail(msg)
                

# find top 30 coin w.r.t to market cap and rank them in terms of 7day momentum, have to plug in your code here.
# test list
  def main(self):
      start_time = time.time()
      api_ready, msg = testapi.test_api_key(self.client, BinanceAPIException)
      msg1=testapi.server_status(self.client)
      msg=str(msg)+'  '+str(msg1)
      mail.send_mail(msg)
    #load date to get data to compute indicators
    # get trading universe w.r.t top 10coins by marketcap now            
      p1 = Thread(target=self.buy)
      p2 = Thread(target=self.sell)    
      p1.start()
      p2.start()
      p1.join()
      p2.join()
      if len(self.coins_bought):
        p.pnl(self.coins_bought)
      print("MP--- %s seconds for single---" % (time.time() - start_time))
# since we need only 100 days data to compute our indicators, we find the dates to do so
    
