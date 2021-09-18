from sys import exit
import yaml
import argparse
import os
from binance.client import Client
import coinmarketcapapi


def load_config(file):
    try:
       if os.path.isfile(file):
          with open(file) as file:
              return yaml.load(file)
    except FileNotFoundError as fe:
        exit(f'Could not find {file}')
    
    except Exception as e:
        exit(f'Encountered exception...\n {e}')

def load_correct_creds(creds):
    try:

        return creds['prod']['access_key'], creds['prod']['secret_key']
    
    except TypeError as te:
        message = 'Your credentials are formatted incorectly\n'
        message += f'TypeError:Exception:\n\t{str(te)}'
        exit(message)
    except Exception as e:
        message = 'oopsies, looks like you did something real bad. Fallback Exception caught...\n'
        message += f'Exception:\n\t{str(e)}'
        exit(message)

def load_cmccreds(creds):
    try:

        return creds['prod']['crypto_key']
    
    except TypeError as te:
        message = 'Your credentials are formatted incorectly\n'
        message += f'TypeError:Exception:\n\t{str(te)}'
        exit(message)
    except Exception as e:
        message = 'oopsies, looks like you did something real bad. Fallback Exception caught...\n'
        message += f'Exception:\n\t{str(e)}'
        exit(message)


def main():
  parsed_creds = load_config('creds.yml')
  access_key, secret_key = load_correct_creds(parsed_creds)
  client = Client(access_key, secret_key)
  return client

def cmc_main():
  parsed_creds = load_config('creds.yml')
  access_key = load_cmccreds(parsed_creds)
  client = coinmarketcapapi.CoinMarketCapAPI(access_key)
  return client
