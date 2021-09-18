import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import sqlalchemy as db
import pymysql
import multiprocess as mp


 
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

end=434
start=0

def scrapper_crypto(b):
    a1=pd.DataFrame();    
    # remove empty string in dict    
    for i in range(0,434):
        b[i]={k: v for k, v in b[i].items() if v}        
    for j in range(0,434):
        for i in list(b[j].keys()):
            ab=pd.DataFrame()
            ab['c']=b[j][i][0]
            ab['d']=b[j][i][1]
            ab['key']=i
            a1 = a1.append(ab, ignore_index = True)
    a1['d']=a1['d'].str[3:]
    a1['key']=a1.key.str.extract('(\d+)')
    a1['key']=pd.to_datetime(a1['key'])    
    a1 = a1.sort_values(by='key')
    az= a1.groupby('c').max()
    az=az.reset_index()
    az.rename(columns={'c':'Crypto','key':'max','d':'name'},inplace=True)
    azz= a1.groupby('c').min()
    azz=azz.reset_index()
    azz.rename(columns={'c':'crypto','key':'min','d':'Name'},inplace=True)
    az=pd.concat([az,azz],axis=1,join='inner')
    del(az['crypto'])
    del(az['name'])
    tableName   = "crypto"
    dataframe_to_sql(az,tableName,sql_connector)

def scrapee(url):
    def dup_utility(ab):
        Crypto = []
        for i in ab:
            if i not in Crypto:
                Crypto.append(i)    
        return Crypto

    import time
    b={}
    Crypto=[];Crypto1=[];
    time.sleep(10)
    print(url)
    import requests
    p=requests.get(url)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(p.content,'html.parser')
    z2=soup.find_all('td', class_="cmc-table__cell cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__symbol")
    z1=soup.find_all('td', class_="cmc-table__cell cmc-table__cell--sticky cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__name")
    for z in z2:
        Crypto.append(z.div.text)
    for z in z1:
        Crypto1.append(z.div.text)    
    crypto=dup_utility(Crypto)
    crypto1=dup_utility(Crypto1)
    b[url]=[crypto,crypto1]
    return b    

def generate_urls():
    all_urls=[]
    a=2013; d=4; c=21; base_url='https://coinmarketcap.com/historical/' ; 
    d= datetime.datetime(a,d,c)
    for i in range(0,434):
        d=d+timedelta(7)
        d1=d.strftime('%Y%m%d')
        url=base_url+d1+'/'        
        all_urls.append(url)
    return all_urls     

all_urls=generate_urls()    

with mp.Pool(1) as pool:
    a=pool.map(scrapee, all_urls)
