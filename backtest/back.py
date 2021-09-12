import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import coinmarketcapapi 
from cryptocmd import CmcScraper
import random
import statistics
from scrape import sql_to_dataframe
from datetime import timedelta


# Data obtained from Binance on 2020-02-04
# Read data with 1 day ticks, it will be used to compute indicators # you can change here for ETH
# Read data with 1 hour ticks, it will be used to simulate price monitoring
 #To find no of Stoploss hit

def getdata():
    Crypto=sql_to_dataframe()
    Crypto['max']=Crypto['max']+timedelta(6)
    Crypto=Crypto.drop(143)
    
    for i in range(len(Crypto)-1,-1,-1):    
        scraper = CmcScraper(Crypto['Crypto'].iloc[i])
        df_1d = scraper.get_dataframe()
        df_1d=df_1d.sort_values(by='Date')
        df_1d['diff']=(df_1d['Date'].diff())
        df_1d['diff']=pd.to_numeric(df_1d['diff'].dt.days, downcast='float')
        if df_1d['Date'].iloc[-1] >= Crypto['max'].iloc[i] and df_1d['Date'].iloc[0] <= Crypto['min'].iloc[i] and df_1d['diff'].iloc[1:].sum() <= (len(df_1d)-1) and len(df_1d) > 110 :
            print(i)
        else:
            print(len(df_1d)-1)
            print(df_1d['diff'].iloc[1:].sum())
            Crypto=Crypto.drop(Crypto['id'].iloc[i])
        return Crypto

def utlity_format(Crypto,i):    
    Crypto['min'].iloc[i]=Crypto['min'].iloc[i].strftime('%d-%m-%Y')
    Crypto['max'].iloc[i]=Crypto['max'].iloc[i].strftime('%d-%m-%Y')       
    return Crypto

def ubound(df,w):
    bound=df.shift(1).rolling(window=w).max()
    return bound

def lbound(df,w):
    bound=df.shift(1).rolling(window=w).min()
    return bound


def crossover(d,d1):
    d2=np.where(d1>d,1,0)
    d3=np.where(d>d1,-1,0)
    d4=d2+d3
    return d4

def stablecoins(z):
    index=[]
    for i in range(len(z)):
        if 'stablecoin' in z[i]:
            index.append(i)
    return index

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def vis(Crypto):
    for Crypto in Crypto:
        scraper = CmcScraper(Crypto)
        df_1d = scraper.get_dataframe()
        df_1d=df_1d.sort_values(by='Date')
        
        df_1d=indicator(df_1d,crossover,ubound,lbound)
        df_1d["Index"]=np.arange(0,df_1d.shape[0])
        plt.scatter(df_1d[ 'Index'],df_1d['Close'],label="Close",color='blue')
        plt.plot(df_1d['Index'],df_1d['upper_bound'],label="Up",color='red')
        plt.plot(df_1d['Index'],df_1d['lower_bound'],label="low",color='green')
        plt.plot(df_1d['Index'],df_1d['50ma'],label="fastma",color='black')
        plt.plot(df_1d['Index'],df_1d['100ma'],label="slowma",color='yellow')
        plt.legend()
        plt.title(Crypto)
        plt.show()

def indicator(df_1d,crossover,ubound,lbound):
    df_1d['N']=ta.atr(df_1d.High.astype(float), df_1d.Low.astype(float), df_1d.Close.astype(float), length=20, mamode='sma', talib=None, drift=None, offset=None, )
    # Compute upper and lower bounds based on Turtle Algorithm
    df_1d['N']=df_1d['N'].shift(1)
    df_1d["50ma"]=ta.tema(df_1d['Close'], length=50, talib=None, offset=None)
    df_1d["100ma"] = ta.tema(df_1d['Close'], length=100, talib=None, offset=None)
    df_1d['pos']=crossover(df_1d['100ma'],df_1d['50ma']) 
    df_1d["upper_bound"] = ubound(df_1d["Close"],50)
    df_1d["lower_bound"] = lbound(df_1d["Close"],20)
    return df_1d


def signal(Crypto):
    success_history = []  # list to keep successful positions
    failure_history = []  # list to keep failed positions
    for j in range(len(Crypto)):
        
        scraper = CmcScraper(Crypto['Crypto'].iloc[j])
        df_1d = scraper.get_dataframe()
        df_1d=df_1d.sort_values(by='Date')   
        positions = []# list to keep current positions
        stop_loss = 0
        df_1d=indicator(df_1d,crossover,ubound,lbound)
        df_1d=df_1d.loc[(df_1d['Date'] >= Crypto['min'].iloc[j])&(df_1d['Date'] <= Crypto['max'].iloc[j])]
    
        for i in range(
                1 , df_1d.shape[0]
                ):
        
                
                if (
                    df_1d["Close"].iloc[i] > df_1d["upper_bound"].iloc[i] 
                    and df_1d['pos'].iloc[i] == 1 and len(positions) < 2 ):
                    
                    date = df_1d.Date.iloc[i]
                    # We will use average price from the current ticker
                    if (i != df_1d.shape[0] - 1):
                        price = ((df_1d["Close"].iloc[i+1] + df_1d["Open"].iloc[i+1]) / 2.0)            
                        stop_loss = price - 3.0 * df_1d["N"].iloc[i]  # set stop loss
                        
                        
                        positions += [{ "date": date, "price": price ,"crypto":Crypto['Crypto'].iloc[j] ,"obj":"Long", }]
                        
                                
                elif (stop_loss != 0 and stop_loss < (positions[0]['price']-3.0 * df_1d["N"].iloc[i]) and len(positions) > 0):
                    stop_loss = positions[0]['price']-3.0 * df_1d["N"].iloc[i]
                    
            # Check to close position
                elif len(positions) > 0 and (
                        df_1d["Close"].iloc[i]
                        < df_1d["lower_bound"].iloc[i]  # we are lower than lower bound
                        or df_1d["Close"].iloc[i] < stop_loss  # we are lower than stop loss
                        or i== df_1d.shape[0] - 1 or df_1d["pos"].iloc[i] == -1  # the end of simulation and we want to close all positions
                        ) : 
           
                       
                        if(i!= df_1d.shape[0] - 1):
                            price = ((df_1d["Close"].iloc[i+1] + df_1d["Open"].iloc[i+1]) / 2.0)            
                        else:
                            price=df_1d["Close"].iloc[i]
                        stop_loss = 0.0
                        if positions[-1]["price"] < price:                
                            for p in positions:
                                success_history += [
                                    {
                                        "date": [p["date"], df_1d.Date.iloc[i]],
                                        "price": [p["price"], price],
                                        "crypto":[p["crypto"] ,Crypto['Crypto'].iloc[j]],
                                        "obj":[p["obj"], "sell"],
                                        
                                    }
                                ]
                        else:
                            for p in positions:
                                failure_history += [
                                    {
                                        "date": [p["date"], df_1d.Date.iloc[i]],
                                        "price": [p["price"], price],
                                        "crypto":[p["crypto"] ,Crypto['Crypto'].iloc[j]],
                                        "obj":[p["obj"], "sell"],
                                    }
                        ]
                        positions = []
    
    return   success_history,  failure_history            
              

def portfolio(success_history,failure_history,capital):
    pp=[];c=[];crypto=0;fees=0.001; price_changes = []; dup=capital
    history = success_history + failure_history
    history.sort(key=lambda history: history['date'])
    a1=pd.DataFrame();a2=pd.DataFrame()
    h=history[0]
    for i in range(len(h['date'])):
        for k in h:    
            a=list(map(lambda coin: coin[k][i],history ))
            a2[k]=a
        if len(a2)==0:
            a1=a2
        else:    
            a1 = a1.append(a2, ignore_index = True)
    a1 = a1.sort_values(by='date')
    
    for i in range(len(a1)):
        if a1['obj'].iloc[i]=='Long' and capital > 12:
            pos = capital * 0.1
            capital = capital - pos
            crypto = np.round((pos * (1.0 - fees)) / a1['price'].iloc[i], 2)
            pos = 0
            pp+=[{'qty':crypto, 'price':a1['price'].iloc[i],'crypto':a1['crypto'].iloc[i]}]
        elif a1['obj'].iloc[i]=='sell':
            for j in range(len(pp)):
                if pp[j]['crypto']==a1['crypto'].iloc[i]:
                    pos = (pp[j]['qty'] * a1['price'].iloc[i] * (1 - fees))
                    n=np.round(((a1['price'].iloc[i] - pp[j]['price'])/ pp[j]['price'])*100,2,)
                    price_changes.append(n)
                    capital=capital + pos
                    dup=dup+(pos-pp[j]['price']*pp[j]['qty'])
                    c.append(dup)
                    pos = 0; crypto=0
                    pp.pop(j)
                    break
    index=np.arange(0,len(c))
    plt.plot(index,c,label="equity curve")
    return c,price_changes


def montecarlo(z):
    r=0 ; prob=0
    b1=[];dummy=[];mdd=[]
    monte=10000
    for j in range(monte):
        dummy=[]
        capital=initial_capita1
        pos=capital*0.1
        capital = capital - pos
        p1=np.random.choice(z,len(z),replace=True)
        for i in range(len(p1)):
            if capital > 50000 :
                capital=capital+((pos*p1[i])/100)
                dummy.append(capital)
            else:
                r+=1
                break
        if capital < initial_capita1:
            prob+=1
        Roll_Max = pd.Series(p1).cummax()
        Daily_Drawdown = (pd.Series(p1)/Roll_Max - 1.0)*100
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        mdd.append(Max_Daily_Drawdown.max())
        
        b1.append(capital)
        
    b1.sort()    
    median_mdd=statistics.median(mdd)
    Median_profit= statistics.median([x - initial_capita1 for x in b1])
    Median_returns= statistics.median([((x - initial_capita1)/initial_capita1)*100 for x in b1])
    risk_of_ruin=(r/(monte)*100)
    returns_dd = abs(Median_returns/median_mdd)
    probabiltiy_of_profit=( 1- prob/monte)*100
    return median_mdd,Median_profit,Median_returns,risk_of_ruin,returns_dd,probabiltiy_of_profit
#function to calculate Sharpe Ratio - Risk free rate element excluded for simplicity
def sharpe_ratio(return_series, N=365, rf=0.05):
    mean = statistics.mean(return_series) * N -rf
    sigma = statistics.stdev(return_series) * np.sqrt(N)
    return mean / sigma

def sortino_ratio(series, N=365, rf=0.05):
    serie=[]
    mean = statistics.mean(series) * N -rf
    for i in series :
        if i < 0 :
            serie.append(i)
    std_neg = statistics.stdev(serie)*np.sqrt(N)
    return mean/std_neg

def max_drawdown(c):
        Roll_Max = pd.Series(c).cummax()
        Daily_Drawdown = (pd.Series(c)/Roll_Max - 1.0)*100
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        ma = Max_Daily_Drawdown.min()
        avg=statistics.mean(Max_Daily_Drawdown)
        return ma, avg

def stats(Capital,percent_change,success_history,failure_history):
    success_rate = 0
    scraper = CmcScraper('BTC')
    df_1d = scraper.get_dataframe()
    
    start_date=df_1d['Date'].iloc[-1]
    print("Start date of simulation", start_date,'\n') 
    
    end_date=df_1d['Date'].iloc[-0]
    print("end date of simulation", end_date,'\n') 
  
    no_of_days=(end_date - start_date)
    print("end date of simulation", no_of_days,'\n')
    
    years= round(no_of_days / np.timedelta64(1,'Y'),0)
    
    CAGR = (Capital[-1]/initial_capita1)**(1/years)-1
    
    print ('Your investment had a CAGR of {:.2%} '.format(CAGR))
    
    no_of_trades=len(success_history) + len(failure_history)
    print("total number of trades taken", no_of_trades,'\n') 
    
    equity_final=Capital[-1]
    print("the final equity ", equity_final, "\n") 
    
    equity_max=max(Capital)
    print("the max equity ", equity_max,'\n') 
    
    return_percentage = ((Capital[-1] - initial_capita1)/initial_capita1)*100
    print("the return percentage of this stratergy ", return_percentage,'\n') 
    
    btc_return_percentage = ((df_1d["Close"].iloc[0] - df_1d["Close"].iloc[-1])/df_1d["Close"].iloc[-1])*100
    print("the return if we held bitcoin all the way ", btc_return_percentage,'\n') 
    
    volatity = statistics.stdev(percent_change)
    print("volatity of your stratergy", volatity,'\n') 
    
    sharpe = sharpe_ratio(percent_change)
    print("sharpe_ratio of your stratergy", sharpe,'\n') 

    sortino = sortino_ratio(percent_change) 
    print("sortino_ratio of your stratergy", sortino,'\n') 

    max_dd , avg_dd =max_drawdown(Capital)
    print("max_drawdown of your stratergy", max_dd,'\n') 
    print("avg_drawdown of your stratergy", avg_dd,'\n') 
    
    calmars = statistics.mean(percent_change)*365/abs(max_dd)
    print("calmars_ratio of your stratergy", calmars,'\n') 
    
    success_rate = len(success_history) / (len(failure_history) + len(success_history))
    print("Success rate", success_rate,'\n') 
    
    big_win=max(percent_change)
    print("Big win", big_win,'\n')
    
    worst_loss=min(percent_change)
    print("Worst loss", worst_loss,'\n')
    
    median_mdd,Median_profit,Median_returns,risk_of_ruin,returns_dd,probabiltiy_of_profit=montecarlo(percent_change)
    print("montecarlo simulations of median drawdown", median_mdd,'\n')
    print("montecarlo simulations of median profits", Median_profit,'\n')
    print("montecarlo simulations of median returns", Median_returns,'\n')
    print("montecarlo simulations for chances of ruin ", risk_of_ruin,'\n')
    print("montecarlo simulations for risk to reward ratio", returns_dd,'\n')
    print("montecarlo simulations for probability of winning", probabiltiy_of_profit,'\n')
    
    p=[] ; l=[]
    
    for i in percent_change:
        if i >= 0:
            p.append(i)
        else:
            l.append(i)
    
    avgwl=statistics.mean(p) /( -statistics.mean(l))
    print("The average win/loss of our stratergy ", avgwl,'\n')
    
    appt=(statistics.mean(p)*(len(success_history) / (len(failure_history) + len(success_history)))-( -statistics.mean(l)*(len(failure_history) / (len(failure_history) + len(success_history)))))
    print("The average probability per trade", appt,'\n')
    
    
    
    
Crypto=getdata()
global initial_capita1
initial_capita1=100000
success_history,failure_history=signal(Crypto)
Capital,  percent_change= portfolio(success_history, failure_history,initial_capita1)    
stats(Capital,percent_change,success_history,failure_history)

plt.hist(Capital)
# Show plot
plt.show()
