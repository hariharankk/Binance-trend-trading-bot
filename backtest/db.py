import numpy as np
import requests
import pandas as pd
from cryptocmd import CmcScraper
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import datetime
from datetime import date, timedelta
import time 
import requests
import sqlalchemy as db
import pymysql

def dup_utility(ab):
    Crypto = []
    for i in ab:
        if i not in Crypto:
            Crypto.append(i)    
    return Crypto

 
def sql_connector():
    sqlEngine       = db.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                               format('root', 'asshole1099','127.0.0.1', 'crypto'), pool_recycle=3600)
    dbConnection    = sqlEngine.connect()

    return dbConnection



def dataframe_to_sql(df,tableName,sql_connector):
    dbConnection=sql_connector()     

    try:
        df.to_sql(tableName, dbConnection, if_exists='replace',index_label='id');
    
    except Exception as ex:   
        print(ex)
    
    else:
        print("Table %s created successfully."%tableName);   

    finally:
        dbConnection.close()


def sql_to_dataframe():
    dbConnection=sql_connector()     
    
    frame  = pd.read_sql("select * from crypto", dbConnection);

    pd.set_option('display.expand_frame_repr', False)

    return(frame)

def scrapper_crypto():
    a=2013; d=4; c=28; url='https://coinmarketcap.com/historical/' ; b={}; a1=pd.DataFrame();
    d= datetime.datetime(a,d,c)
    d1=d.strftime('%Y%m%d')
    url=url+d1+'/'
    
    for i in  range(434):
        Crypto=[]
        while True: 
            p=requests.get(url)
            if p.status_code == 200:
                break
        soup = BeautifulSoup(p.content,'html.parser')
        z2=soup.find_all('td', class_="cmc-table__cell cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__symbol")
        for z in z2:
            Crypto.append(z.div.text)
        crypto=dup_utility(Crypto)
        url='https://coinmarketcap.com/historical/'
        b[d]=crypto
        d=d+timedelta(7)
        d1=d.strftime('%Y%m%d')
        url=url+d1+'/'
        
    # remove empty string in dict    
    b={k: v for k, v in b.items() if v}        
    for i in list(b.keys()):
        ab=pd.DataFrame()
        ab['c']=b[i]
        ab['key']=i
        a1 = a1.append(ab, ignore_index = True)
    a1 = a1.sort_values(by='key')
    az= a1.groupby('c').max()
    az=az.reset_index()
    az.rename(columns={'c':'Crypto','key':'max'},inplace=True)
    azz= a1.groupby('c').min()
    azz=azz.reset_index()
    azz.rename(columns={'c':'crypto','key':'min'},inplace=True)
    az=pd.concat([az,azz],axis=1,join='inner')
    del(az['crypto'])
    tableName   = "crypto"
    dataframe_to_sql(az,tableName,sql_connector)

