import pandas as pd
import numpy as np
import requests
from cryptocmd import CmcScraper
from datetime import timedelta
import pymongo
import json


class scraper(object):
    def __init__(self):
      self.collection = self.mongo()
      self.no_data = pd.DataFrame()
      self.missing_data = pd.DataFrame()
      self.Crypto = self.get_coinmarketcap_data()#pd.DataFrame()
      

    def get_coinmarketcap_data(self):
      query = self.collection.find_one({"index":"First_Name"},{"data":1,"_id":0})
      coins=pd.DataFrame(json.loads(query['data']),columns = json.loads(query['data']).keys())
      coins=coins.reset_index()
      return coins

    
    def mongo(self):
      # uri (uniform resource identifier) defines the connection parameters 
      uri = 'mongodb+srv://crypto:asshole1099@cluster0.vjecl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'
      # start client to connect to MongoDB server 
      client = pymongo.MongoClient( uri )
      collection = client.Crypto.coins 
      return collection


    def cmc_scraper(self,crypto):
      scraper = CmcScraper(crypto)
      df_1d = scraper.get_dataframe()
      df_1d=df_1d.sort_values(by='Date')
      return df_1d 

    def utlity_format(self):    
        self.Crypto['max']=pd.to_datetime(self.Crypto['max'],format='%Y-%m-%d %H:%M:%S')
        self.Crypto['min']=pd.to_datetime(self.Crypto['min'],format='%Y-%m-%d %H:%M:%S')       
        return self.Crypto

    def difference(self,df_1d):
        df_1d['diff']=(df_1d['Date'].diff())
        df_1d['diff']=pd.to_numeric(df_1d['diff'].dt.days, downcast='float')
        return df_1d

    def json_to_df(self,df):
      df = df.to_json()
      return df


    def getdata(self):
        self.Crypto = self.get_coinmarketcap_data()
        self.utlity_format()
        self.Crypto['Crypto'] = self.Crypto['Crypto'].replace(['STRAT'],'STRAX')
        self.Crypto['Crypto'] = self.Crypto['Crypto'].replace(['NANO'],'XNO')
        self.Crypto['Crypto'] = self.Crypto['Crypto'].replace(['GNT'],'GLM')
        self.Crypto[self.Crypto.Crypto!='MEXC']

        

        for i in range(len(self.Crypto)-1,-1,-1):    
            try:
                
                df_1d = self.cmc_scraper(self.Crypto['Crypto'].iloc[i])
                df_1d=self.difference(df_1d)
                if df_1d['Date'].iloc[-1] >= self.Crypto['max'].iloc[i] and df_1d['Date'].iloc[0] <= self.Crypto['min'].iloc[i] and (df_1d['diff'].iloc[1:].sum() - (len(df_1d)-1)) <= 2 and len(df_1d) > 110 :
                    del df_1d['diff']
                    df_1d=self.json_to_df(df_1d)
                    self.collection.insert_one({"index":self.Crypto['Crypto'].iloc[i],"data":df_1d})
                else:
                    if df_1d['diff'].iloc[1:].sum() > (len(df_1d)-1) and df_1d['Date'].iloc[-1] >= self.Crypto['max'].iloc[i] and df_1d['Date'].iloc[0] <= self.Crypto['min'].iloc[i] and len(df_1d) > 110:
                        del df_1d['diff']
                        df=df_1d.loc[(df_1d['Date'] >= self.Crypto['min'].iloc[i])&(df_1d['Date'] <= self.Crypto['max'].iloc[i])]
                        df=self.difference(df)
                        if df['diff'].iloc[1:].sum() <= (len(df)-1):
                          df_1d=df_1d=self.json_to_df(df_1d)
                          self.collection.insert_one({"index":self.Crypto['Crypto'].iloc[i],"data":df_1d})
                        else:
                          self.missing_data=self.missing_data.append(self.Crypto.iloc[i],ignore_index=True)
                          self.Crypto=self.Crypto.drop(self.Crypto['id'].iloc[i])
                    else:
                      self.no_data=self.no_data.append(self.Crypto.iloc[i],ignore_index=True)
                      self.Crypto=self.Crypto.drop(self.Crypto['id'].iloc[i])
            except ConnectionError as e:
                print(e)
        json_Crypto=self.Crypto.to_json()
        self.collection.insert_one({"index":"Name" , "data":json_Crypto})        
        return self.collection
