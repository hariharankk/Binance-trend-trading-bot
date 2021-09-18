import json
import os
def json1():
    coins_bought = {}    
    coins_bought_file_path = 'coins_bought.json'
    if os.path.isfile(coins_bought_file_path):
        with open(coins_bought_file_path) as file:
            coins_bought = json.load(file)
    return coins_bought,coins_bought_file_path  

def update_porfolio(orders, trades):
    '''add every coin bought to our portfolio for tracking/selling later'''
    coinsbought,coins_bought_file_path=json1()
    for coin in orders:
        # add existing coin
       if coin in coinsbought :       
              vol=float(coinsbought[coin]['volume'])
              coinsbought[coin]['symbol'] = orders[coin][0]['symbol']
              coinsbought[coin]['orderid'] = orders[coin][0]['orderId']
              coinsbought[coin]['bought_at'] = trades[coin][0]['price']
              coinsbought[coin]['volume'] = float(orders[coin][0]['origQty'])+vol
              coinsbought[coin]['count'] = 1 + coinsbought[coin]['count']    
                                        
            
       else:
           #add if coin in new
              coinsbought[coin] = {
                    'symbol': orders[coin][0]['symbol'],
                    'orderid': orders[coin][0]['orderId'],
                    'timestamp': orders[coin][0]['time'],
                    'bought_at': trades[coin][0]['price'],
                    'volume': float(orders[coin][0]['origQty']),
                    'count':1
                    }
            
            # save the coins in a json file in the same directory
       with open(coins_bought_file_path, 'w') as file:
            json.dump(coinsbought, file, indent=4)
