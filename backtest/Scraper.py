import numpy as np
import time
import pandas as pd
import datetime
from datetime import date, timedelta
import multiprocess as mp
import requests
from bs4 import BeautifulSoup
import random
import re
import pymongo
import json


class scraperr(object):
  def __init__(self):
    self.all_urls = []
    self.end = 460
    self.start = 0
    self.Crypto_list=[]
  
  def mongo(self):
    uri = 'mongodb+srv://crypto:xxxxxx@cluster0.vjecl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'
    # start client to connect to MongoDB server 
    client = pymongo.MongoClient( uri )
    collection = client.Crypto.coins 
    return collection

  def ultity(self,df,func):  
    df=pd.DataFrame(df)
    if func=='max':
      dummy=df.groupby('crypto').max()
    elif func=='min':
      dummy=df.groupby('crypto').min()
    dummy=dummy.reset_index()  
    dummy=dummy.reset_index()
    dummy.rename(columns={'index':'id'},inplace=True)
    return dummy


  def GET_UA(self):
    uastrings = ["Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.72 Safari/537.36",\
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",\
                "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 Safari/537.85.10",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",\
                "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",\
                "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36"\
                ]
 
    return random.choice(uastrings)


  def date_format (self,url):  
    temp = re.findall(r'\d+', url)
    year = int(temp[0][0:4]); month = int(temp[0][4:6]); day = int(temp[0][6:8])
    start_date= datetime.datetime(year,month,day)
    end_date=start_date+timedelta(7)
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
    return start_date,end_date

  def generate_urls(self):
    year=2013; month=4; day=21; base_url='https://coinmarketcap.com/historical/'; 
    url_date= datetime.datetime(year,month,day)
    for i in range(self.start,self.end):
        url_date=url_date+timedelta(7)
        string_url_date=url_date.strftime('%Y%m%d')
        url=base_url+string_url_date+'/'        
        self.all_urls.append(url)
    return self.all_urls     

  def scrapee(self):
    self.all_urls=self.generate_urls()
    for i in range(len(self.all_urls)):
      Crypto=[];
      time.sleep(2)
      headers = {'User-Agent': self.GET_UA()}
      p=requests.get(self.all_urls[i], headers=headers ).text
      soup = BeautifulSoup(p,'html.parser')
      z1=soup.find_all('td', class_="cmc-table__cell cmc-table__cell--sticky cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__name")
      for z in z1: 
        z2=z.find_all('a', class_="cmc-table__column-name--symbol cmc-link")
        for z3 in z2:
          Crypto.append(z3.text)
      start_date, end_date =self.date_format(self.all_urls[i])
      self.Crypto_list.append({'crypto':Crypto, 'startdate':start_date, 'enddate':end_date})
    return self.Crypto_list  

  def date_formating(self):
    self.Crypto_list=self.scrapee()
    data = pd.DataFrame(self.Crypto_list)
    Start_datafield=[]; end_datafield=[]
    for i in range(len(data)):
      for j in range(len(data['crypto'].iloc[i])):
        Crypto_list=data['crypto'].iloc[i]
        Start_datafield.append({'crypto':Crypto_list[j],'startdate':data['startdate'].iloc[i]})
        end_datafield.append({'crypto':Crypto_list[j],'enddate':data['enddate'].iloc[i]})    

    Start_datafield=self.ultity(Start_datafield,'min')
    end_datafield=self.ultity(end_datafield,'max')
    final_data=pd.merge(Start_datafield,end_datafield,on='crypto')
    del(final_data['id_y'])
    final_data.rename(columns={'id_x':'id','crypto' : 'Crypto','startdate':'min','enddate':'max'},inplace=True)
    collection=self.mongo()
    json_Crypto=final_data.to_json()
    collection.insert_one({"index":"First_Name" , "data":json_Crypto})        
    return final_data

