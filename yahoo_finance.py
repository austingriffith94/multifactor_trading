# Austin Griffith
# Yahoo Finance Web Scraper
# Python 3.6.5

# code is based on yahoo scraper found at
# https://www.scrapehero.com/scrape-yahoo-finance-stock-market-data/

from lxml import html  
import requests
import json
from collections import OrderedDict
from time import sleep
import pandas as pd
import urllib3

#%%
def parse(ticker):
    urllib3.disable_warnings()
    url = "http://finance.yahoo.com/quote/%s?p=%s"%(ticker,ticker)
    response = requests.get(url,verify=False)
    print ("Parsing %s"%(url))
    sleep(0.5)
    parser = html.fromstring(response.text)
    summary = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
    summary_data = OrderedDict()
    other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
    summary_json_response = requests.get(other_details_json_link)
    try:
    	json_loaded_summary =  json.loads(summary_json_response.text)
    	y_Target_Est = json_loaded_summary["quoteSummary"]["result"][0]["financialData"]["targetMeanPrice"]['raw']
    	earnings_list = json_loaded_summary["quoteSummary"]["result"][0]["calendarEvents"]['earnings']
    	eps = json_loaded_summary["quoteSummary"]["result"][0]["defaultKeyStatistics"]["trailingEps"]['raw']
    	datelist = []
    	for i in earnings_list['earningsDate']:
    		datelist.append(i['fmt'])
    	earnings_date = ' to '.join(datelist)
    	for data in summary:
    		rawKey = data.xpath('.//td[contains(@class,"C(black)")]//text()')
    		rawValue = data.xpath('.//td[contains(@class,"Ta(end)")]//text()')
    		tableKey = ''.join(rawKey).strip()
    		tableValue = ''.join(rawValue).strip()
    		summary_data.update({tableKey:tableValue})
    	summary_data.update({'1y Target Est':y_Target_Est,'EPS (TTM)':eps,'Earnings Date':earnings_date,'ticker':ticker,'url':url})
    	return(summary_data)
    except:
    	print ("Failed to parse json response")
    	return({})
        
def yahoo_data(tickerlist):
    mktdata = pd.DataFrame()
    for t in tickerlist['Symbol']:
        bro = parse(t)
        if len(bro) != 0:
            temp = pd.DataFrame([bro]).set_index('ticker')
            mktdata = pd.concat([mktdata,temp])
    return(mktdata)

#%%
tickers = pd.read_csv('tickerlist.csv')
mkt = yahoo_data(tickers)
mkt.to_csv('scrapedYahoo.csv')

    