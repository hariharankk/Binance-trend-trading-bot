import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from cryptocmd import CmcScraper
import random
import statistics
from datetime import timedelta
import import_ipynb
import mongodb
import json
import multiprocessing 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc



class backtesting:
  def __init__(self):
    self.success_history = []  
    self.failure_history = []  
    self.rand_data=[]
    self.rand_out=[]
    self.df_btc=self.get_data('BTC')
    self.capital=100000
    self.price_change=[]

  
  def get_data(self,coin):
    scraper_obj = mongodb.scraper()
    coll = scraper_obj.collection
    query = coll.find_one({"index":coin},{"data":1,"_id":0})
    df=pd.DataFrame(json.loads(query['data']),columns=json.loads(query['data']).keys())
    df['Date']=pd.to_datetime(df['Date'] , unit = 'ms')
    return df


  def ubound(self,df):
        bound=df.shift(2).rolling(window=20).max()
        return bound

  def lbound(self,df):
        bound=df.shift(2).rolling(window=10).min()
        return bound

  def crossover(self,d,d1):
        d2=np.where(d1>d,1,0)
        d3=np.where(d>d1,-1,0)
        d4=d2+d3
        return d4


  def vis(self,df_1d):        
        df_1d["Index"]=np.arange(0,df_1d.shape[0])
        plt.scatter(df_1d[ 'Index'],df_1d['Close'],label="Close",color='blue')
        plt.plot(df_1d['Index'],df_1d['upper_bound'],label="Up",color='red')
        plt.plot(df_1d['Index'],df_1d['lower_bound'],label="low",color='green')
        plt.plot(df_1d['Index'],df_1d['50ma'],label="fastma",color='black')
        plt.plot(df_1d['Index'],df_1d['100ma'],label="slowma",color='yellow')
        plt.legend()
        plt.title("Crypto")
        plt.show()

  def indicator(self,df_1d):
        df_1d['atr']=ta.atr(df_1d.High.astype(float), df_1d.Low.astype(float), df_1d.Close.astype(float), length=20, mamode='sma', talib=None, drift=None, offset=None, )
        # Compute upper and lower bounds based on Turtle Algorithm
        df_1d['N']=df_1d['atr'].shift(2)
        df_1d["50ma"]=ta.ema(df_1d['Close'], length=10, talib=None, offset=None)
        df_1d["100ma"] = ta.ema(df_1d['Close'], length=25, talib=None, offset=None)
        df_1d['pos']=self.crossover(df_1d['100ma'],df_1d['50ma']) 
        df_1d["upper_bound"] = self.ubound(df_1d["Close"])
        df_1d["lower_bound"] = self.lbound(df_1d["Close"])
        df_1d['rsi']=ta.rsi(df_1d['Close'], length=14, scalar=100, drift=1)
        df_1d['returns'] = np.log(df_1d['Close'] / df_1d['Close'].shift(1))
        df_1d['Volatility'] = df_1d['returns'].rolling(window=50).std() * np.sqrt(50)
        df_1d['willr']=ta.willr(df_1d['High'], df_1d['Low'], df_1d['Close'], length=14)
        df_1d['200ma']=ta.sma(df_1d['Close'], length=200)
        df_1d['obv']=ta.obv(df_1d['Close'], df_1d['Volume'])
        df_1d['Chop']=ta.chop(df_1d['High'], df_1d['Low'], df_1d['Close'])
        return df_1d


  def signal(self, Crypto):
    for j in range(len(Crypto)):  
            df_1d = self.get_data(Crypto['Crypto'].iloc[j])
            positions = []# list to keep current positions
            stop_loss = 0
            df_1d=self.indicator(df_1d)
            df_1d=df_1d.loc[(df_1d['Date'] >= Crypto['min'].iloc[j])&(df_1d['Date'] <= Crypto['max'].iloc[j])]

            for i in range(
                    1 , df_1d.shape[0]
                    ):
            
                    date = df_1d.Date.iloc[i]
                    if (
                        df_1d["Close"].iloc[i-1] > df_1d["upper_bound"].iloc[i-1] 
                        and df_1d['pos'].iloc[i-1] == 1 and len(positions) < 2 ):
                        
                        
                        # We will use average price from the current ticker
                        if (date != df_1d.Date.iloc[df_1d.shape[0]-1]):
                            price = ((df_1d["Close"].iloc[i] + df_1d["Open"].iloc[i]) / 2.0)            
                            stop_loss = price - 3.0 * df_1d["N"].iloc[i]  # set stop loss
                            
                            self.df_btc['ret']=self.df_btc['Close'].astype(float).pct_change()
                            if self.df_btc['ret'].iloc[np.where(df_1d['Date'].iloc[i-1]==self.df_btc['Date'])][0] > 0:
                              beta=1
                            else:
                              beta=0                              
                            
                            positions += [{ "date": date, "price": price, "crypto": Crypto['Crypto'].iloc[j] , "obj":"Long", }] 
                            self.rand_data+=[{"datebought": date, "Close":df_1d["Close"].iloc[i-1], "High":df_1d["High"].iloc[i-1] ,"Low":df_1d["Low"].iloc[i-1] , "crypto": Crypto['Crypto'].iloc[j] , "Open" : df_1d["Open"].iloc[i-1] , "marketcap" : df_1d["Market Cap"].iloc[i-1] , "atr":df_1d["atr"].iloc[i-1], "rsi":df_1d["rsi"].iloc[i-1], "200ma":df_1d["200ma"].iloc[i-1], "Volume":df_1d["Volume"].iloc[i-1], "obv":df_1d["obv"].iloc[i-1], "Volatility":df_1d["Volatility"].iloc[i-1], "100ma" : df_1d["100ma"].iloc[i-1] , "50ma" : df_1d["50ma"].iloc[i-1]  , "upperbound" : df_1d["upper_bound"].iloc[i-1] ,  "lowerbound" : df_1d["lower_bound"].iloc[i-1] , "willr":df_1d["willr"].iloc[i-1] , "chop" : df_1d["Chop"].iloc[i-1], 'beta':beta  }] 
                                    
                    elif (stop_loss != 0 and stop_loss < (positions[-1]['price']-3.0 * df_1d["N"].iloc[i]) and len(positions) > 0 and date != df_1d.Date.iloc[df_1d.shape[0]-1] ) :
                        stop_loss = df_1d["Close"].iloc[i]-3.0 * df_1d["N"].iloc[i]
                        
                # Check to close position
                    elif len(positions) > 0 and (
                            df_1d["Close"].iloc[i-1]
                            < df_1d["lower_bound"].iloc[i-1]  # we are lower than lower bound
                            or df_1d["Close"].iloc[i-1] < stop_loss  # we are lower than stop loss
                            or date == df_1d.Date.iloc[df_1d.shape[0]-1] or df_1d["pos"].iloc[i-1] == -1  # the end of simulation and we want to close all positions
                            ) : 
              
                          
                            if(df_1d.Date.iloc[df_1d.shape[0]-1]):
                                price = ((df_1d["Close"].iloc[i] + df_1d["Open"].iloc[i]) / 2.0)            
                            else:
                                price=df_1d["Close"].iloc[i]
                            stop_loss = 0.0
                            if positions[-1]["price"] < price:                
                                for p in positions:
                                    self.rand_out += [{"datesold" : date, "crypto": Crypto['Crypto'].iloc[j], "datebought": p["date"] , 'output':1}] 
                                    self.success_history += [
                                        {
                                            "date": [p["date"], date],
                                            "price": [p["price"], price],
                                            "crypto": [Crypto['Crypto'].iloc[j],Crypto['Crypto'].iloc[j]],
                                            "obj":[p["obj"], "sell"],
                                            
                                        }
                                    ]
                            else:
                                for p in positions:
                                    self.rand_out += [{"datesold" : df_1d.Date.iloc[i], "crypto": Crypto['Crypto'].iloc[j] , "datebought": p["date"] , 'output':0}] 
                                    self.failure_history += [
                                        {
                                            "date": [p["date"], date],
                                            "price": [p["price"], price],
                                            "crypto": [Crypto['Crypto'].iloc[j],Crypto['Crypto'].iloc[j]],
                                            "obj":[p["obj"], "sell"],
                                        }
                            ] 
                            positions = []
        
    return   self.success_history,  self.failure_history           
                
class rfclassifier(backtesting):
  def __init__(self):
    super().__init__()

  def clean_dataset(self,df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]
  

  def randomforestmodel(self):
    data_feature=pd.DataFrame(self.rand_data)
    data_label=pd.DataFrame(self.rand_out)
    df = pd.merge(data_feature, data_label, on=['datebought', 'crypto'], how='left')
    df=self.clean_dataset(df)    
    df['pred'] = 1
    
    actual = df['output']
    pred = df['pred']
    print(classification_report(y_true=actual, y_pred=pred))

    print("Confusion Matrix")
    print(confusion_matrix(actual, pred))

    print('')
    print("Accuracy")
    print(accuracy_score(actual, pred))
    
    X = df.drop(['output','datebought','datesold','crypto','pred'],axis=1)
    y = df['output']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1,bootstrap=True, oob_score=True)
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 3 , 4 , 5, 6 ]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3 , 4, 5]
    
    #Method of selecting samples for training each tree
    
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               }
    grid_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    
    grid_search.fit(X_train, y_train)
    score=grid_search.best_score_

    
    rf_best = grid_search.best_estimator_
    rf_best.fit(X_train, y_train)
    
    
    y_pred_rf = rf_best.predict_proba(X_test)[:, 1]
    y_pred = rf_best.predict(X_test)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print('')
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    imp_df = pd.DataFrame({
        "Varname": X_train.columns,
        "Imp": rf_best.feature_importances_
    })

    imp_df.sort_values(by="Imp", ascending=False)
    imp_df=imp_df.sort_values(by=['Imp'], ascending=False)
    imp_df['cumsum']=np.cumsum(imp_df['Imp'])
    imp_df.reset_index(inplace = True)
    imp_df=imp_df.loc[np.where(imp_df['cumsum']>0.95)[0]]
    columns=list(imp_df['Varname'])
    X.drop(columns, axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    
    rf_best.fit(X_train, y_train)
    y_pred_rf = rf_best.predict_proba(X_test)[:, 1]
    y_pred = rf_best.predict(X_test)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print('')
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    joblib.dump(rf_best, "./random_forest.joblib")
    return imp_df 

class performance(backtesting):
  def __init__(self):
    super().__init__()

  def getdata(self):
    scraper_obj = mongodb.scraper()
    coll = scraper_obj.collection
    query = coll.find_one({"index":"Name"},{"data":1,"_id":0})
    coins=pd.DataFrame(json.loads(query['data']),columns = json.loads(query['data']).keys())
    coins['min']=pd.to_datetime ( coins['min'], unit = 'ms' )
    coins['max']=pd.to_datetime ( coins['max'], unit = 'ms' )
    return coins
  

  def portfolio(self):
    coins=self.getdata()
    super().signal(coins)
    position=[];crypto=0;fees=0.001;  
    history = self.success_history + self.failure_history
    history.sort(key=lambda history: history['date'])
    crypto=pd.DataFrame();dummy1=pd.DataFrame()
    instance=history[0]
    for i in range(len(instance['date'])):
        for k in instance:    
            dummy=list(map(lambda coin: coin[k][i],history ))
            dummy1[k]=dummy
        if len(dummy1)==0:
            crypto=dummy1
        else:    
            crypto = crypto.append(dummy1, ignore_index = True)
    crypto = crypto.sort_values(by='date')
    
    for i in range(len(crypto)):
        if crypto['obj'].iloc[i]=='Long' and self.capital > 12:
            pos = self.capital * 0.1
            self.capital = self.capital - pos
            coin = np.round((pos * (1.0 - fees)) / crypto['price'].iloc[i], 2)
            pos = 0
            position+=[{'qty':coin, 'price':crypto['price'].iloc[i],'crypto':crypto['crypto'].iloc[i]}]
        elif crypto['obj'].iloc[i]=='sell':
            for j in range(len(position)):
                if position[j]['crypto']==crypto['crypto'].iloc[i]:
                    pos = (position[j]['qty'] * crypto['price'].iloc[i] * (1 - fees))
                    dummy=np.round(((crypto['price'].iloc[i] - position[j]['price'])/ position[j]['price']),2,)
                    self.price_change.append(dummy)
                    self.capital=self.capital + pos
                    pos = 0; coin=0
                    position.pop(j)
                    break
    self.stats()                
    return self.price_change

  def sharpe_ratio(self,return_series, N=365, rf=0.05):
      mean = statistics.mean(return_series) * N -rf
      sigma = statistics.stdev(return_series) * np.sqrt(N)
      return mean / sigma

  def sortino_ratio(self,series, N=365, rf=0.05):
      serie=[]
      mean = statistics.mean(series) * N -rf
      for i in series :
          if i < 0 :
              serie.append(i)
      std_neg = statistics.stdev(serie)*np.sqrt(N)
      return mean/std_neg

  def max_drawdown(self,c):
          Roll_Max = pd.Series(c).cummax()
          Daily_Drawdown = (pd.Series(c)/Roll_Max - 1.0)
          Max_Daily_Drawdown = Daily_Drawdown.cummin()
          ma = Max_Daily_Drawdown.min()
          avg=statistics.mean(Max_Daily_Drawdown)
          return ma, avg

  def montecarlo(self):
      counter=0 ; prob=0
      montecapital=[];mdd=[]
      initial_capita1=100000
      monte=10000
      for j in range(monte):
          dummy=[]
          capital=initial_capita1
          pos=capital*0.1
          capital = capital - pos
          random_trade=np.random.choice(self.price_change,len(self.price_change),replace=True)
          for i in range(len(random_trade)):
              if capital > 50000 :
                  capital=capital+((pos*random_trade[i])/100)
                  dummy.append(capital)
              else:
                  counter+=1
                  break
          if capital < initial_capita1:
              prob+=1
          Roll_Max = pd.Series(random_trade).cummax()
          Daily_Drawdown = (pd.Series(random_trade)/Roll_Max - 1.0)*100
          Max_Daily_Drawdown = Daily_Drawdown.cummin()
          mdd.append(Max_Daily_Drawdown.max())
          
          montecapital.append(capital)
          
      montecapital.sort()    
      median_mdd=statistics.median(mdd)
      Median_profit= statistics.median([x - initial_capita1 for x in montecapital])
      Median_returns= statistics.median([((x - initial_capita1)/initial_capita1)*100 for x in montecapital])
      risk_of_ruin=(counter/(monte)*100)
      returns_dd = abs(Median_returns/median_mdd)
      probabiltiy_of_profit=( 1- prob/monte)*100
      return median_mdd,Median_profit,Median_returns,risk_of_ruin,returns_dd,probabiltiy_of_profit


  def stats(self):
      success_rate = 0
      initial_capita1=100000
      
      start_date=self.df_btc['Date'].iloc[0]
      print("Start date of simulation", start_date,'\n') 
      
      end_date=self.df_btc['Date'].iloc[-1]
      print("end date of simulation", end_date,'\n') 
    
      no_of_days=(end_date - start_date)
      print("end date of simulation", no_of_days,'\n')
      
      years= round(no_of_days / np.timedelta64(1,'Y'),0)
      
      CAGR = (self.capital/initial_capita1)**(1/years)-1
      
      print ('Your investment had a CAGR of {:.2%} '.format(CAGR))
      
      no_of_trades=len(self.success_history) + len(self.failure_history)
      print("total number of trades taken", no_of_trades,'\n') 
      
      equity_final=self.capital
      print("the final equity ", equity_final, "\n") 
      
      
      return_percentage = ((self.capital - initial_capita1)/initial_capita1)*100
      print("the return percentage of this stratergy ", return_percentage,'\n') 
      
      btc_return_percentage = ((self.df_btc["Close"].iloc[-1] - self.df_btc["Close"].iloc[0])/self.df_btc["Close"].iloc[0])*100
      print("the return if we held bitcoin all the way ", btc_return_percentage,'\n') 
      
      volatity = statistics.stdev(self.price_change)
      print("volatity of your stratergy", volatity,'\n') 
      
      sharpe = self.sharpe_ratio(self.price_change)
      print("sharpe_ratio of your stratergy", sharpe,'\n') 

      sortino =  self.sortino_ratio(self.price_change) 
      print("sortino_ratio of your stratergy", sortino,'\n') 

      max_dd , avg_dd = self.max_drawdown(self.price_change)
      print("max_drawdown of your stratergy", max_dd,'\n') 
      print("avg_drawdown of your stratergy", avg_dd,'\n') 
      
      calmars = statistics.mean(self.price_change)*365/abs(max_dd)
      print("calmars_ratio of your stratergy", calmars,'\n') 
      
      success_rate = len(self.success_history) / (len(self.failure_history) + len(self.success_history))
      print("Success rate", success_rate,'\n') 
      
      big_win=max(self.price_change)
      print("Big win", big_win,'\n')
      
      worst_loss=min(self.price_change)
      print("Worst loss", worst_loss,'\n')
      
      median_mdd,Median_profit,Median_returns,risk_of_ruin,returns_dd,probabiltiy_of_profit=self.montecarlo()
      print("montecarlo simulations of median drawdown", median_mdd,'\n')
      print("montecarlo simulations of median profits", Median_profit,'\n')
      print("montecarlo simulations of median returns", Median_returns,'\n')
      print("montecarlo simulations for chances of ruin ", risk_of_ruin,'\n')
      print("montecarlo simulations for risk to reward ratio", returns_dd,'\n')
      print("montecarlo simulations for probability of winning", probabiltiy_of_profit,'\n')
      
      p=[] ; l=[]
      
      for i in self.price_change:
          if i >= 0:
              p.append(i)
          else:
              l.append(i)
      
      avgwl=statistics.mean(p) /( -statistics.mean(l))
      print("The average win/loss of our stratergy ", avgwl,'\n')
      
      appt=(statistics.mean(p)*(len(self.success_history) / (len(self.failure_history) + len(self.success_history)))-( -statistics.mean(l)*(len(self.failure_history) / (len(self.failure_history) + len(self.success_history)))))
      print("The average probability per trade", appt,'\n')

class execution(performance,rfclassifier):
  def __init__(self):
    super().__init__()
  def  portfolio(self):
    super().portfolio()



