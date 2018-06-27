# Austin Griffith
# Multifactor Trading
# Python 3.6.5

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime as dt
from pandas_datareader import data as pdr
import statsmodels.api as sm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import shutil

# fixes yahoo api for pandas datareader
import fix_yahoo_finance as yf
yf.pdr_override()

# plot parameters
params = {'legend.fontsize': 20,
          'figure.figsize': (13,9),
         'axes.labelsize': 20,
         'axes.titlesize':25,
         'xtick.labelsize':15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

#%%
# read into ticker directory (which also contains market cap data)
# limits tickers based on desired market cap
# return cleaned directory
def directory(direc,cap):
    newDirec = pd.DataFrame(direc['Symbol'])
    newDirec['ipo'] = direc['IPOyear']
    newDirec['MarketCap'] = direc['MarketCap']
    newDirec = newDirec[newDirec['MarketCap'] >= cap]
    newDirec = newDirec.reset_index(drop=True)
    return(newDirec)

def cci_calc(retdata,win):
    rollingclose = retdata.rolling(window=win,center=False)
    typical = (retdata + rollingclose.max() + rollingclose.min())/3.0
    rollingtyp = rollingclose.mean().fillna(0)
    mad = (np.abs(retdata - rollingtyp)).rolling(window=win,center=False).mean()
    cci = (typical - rollingtyp)/(0.15*mad)
    return(cci)

def rsi_calc(retdata,win):
    delta = retdata.diff()
    dup, ddown = delta.copy(), delta.copy()
    dup[dup < 0] = 0
    ddown[ddown > 0] = 0

    rolup = dup.rolling(window=win,center=False).mean()
    roldown = np.abs(ddown).rolling(window=win,center=False).mean()

    R = 100.0*rolup/(roldown+rolup)
    return(R)

class reader:
    def __init__(self,folder,start,end,winvol=20,winrev=15,winvolume=15,wincci=20,winrsi=15):
        # creates data directory for data
        if os.path.exists(folder) == False:
            os.makedirs(folder)
        self.folder = folder
        self.start = start
        self.end = end
        self.winvol = winvol
        self.winrev = winrev
        self.winvolume = winvolume
        self.wincci = wincci
        self.winrsi = winrsi
        self.maxwindow = max(winvol+1,winrev,winvolume,wincci,wincci)

    def data_dl(self,tick,api='yahoo'):
        # choose api
        if api == 'quandl':
            retcolumn = 'AdjClose'
            volcolumn = 'AdjVolume'
            try:
                data = pdr.DataReader(tick,'quandl',self.start,self.end)
                if data.empty == True:
                    print('Error :',tick)
                    return()
            except:
                print('Error :',tick)
                return()

        elif api == 'yahoo':
            retcolumn = 'Adj Close'
            volcolumn = 'Volume'
            try:
                data = data = pdr.get_data_yahoo(tick,self.start,self.end)
                if data.empty == True:
                    print('Error :',tick)
                    return()
            except:
                print('Error :',tick)
                return()

        const = 2.55*10**-6

        # adjusted returns
        # logarithmic returns
        # volatility
        # momentum
        # reversion
        # rolling average adjusted trade volume
        returns = (data[retcolumn] - data[retcolumn].shift(1))/data[retcolumn].shift(1)
        logret = np.log(const + data[retcolumn]) - np.log(const + data[retcolumn].shift(1))
        volatility = np.log(const + logret.shift(1).rolling(window=self.winvol,center=False).std())
        momentum = np.log(const + data[retcolumn].shift(1)) - np.log(const + data[retcolumn].shift(5))
        reversion = np.log(const + data[retcolumn].shift(self.winrev)) - np.log(const + data[retcolumn].shift(1))
        avgvolume = data[volcolumn].rolling(window=self.winvolume,center=False).mean()

        # Commodity Channel Index (CCI)
        cci = cci_calc(data[retcolumn],self.wincci)

        # Relative Strength Index (RSI)
        rsi = rsi_calc(data[retcolumn],self.winrsi)

        # Moving Average Convergence/Divergence (MACD)
        exp12 = data[retcolumn].ewm(span=12,adjust=True).mean()
        exp26 = data[retcolumn].ewm(span=26,adjust=True).mean()
        macd = exp12 - exp26

        # On Balance Volume (OBV)
        obv = np.multiply(data['Volume'],(~data[retcolumn].diff().le(0)*2 - 1)).cumsum()

        # Append desired variables to dataframe
        data['AvgVolume'] = avgvolume
        data['Returns'] = returns
        data['Vol'] = volatility
        data['Rev'] = reversion
        data['Momentum'] = momentum
        data['CCI'] = cci
        data['MACD'] = macd
        data['OBV'] = obv
        data['RSI'] = rsi

        data = data[self.maxwindow:]

        data.to_csv(self.folder+'/'+tick+'.csv')
        print('Download successful :',tick)

# create master data list
def stockPanel(folder,ticklist):
    master = {}
    for t in ticklist:
        filename = folder+'/'+t+'.csv'
        data = pd.read_csv(filename)
        master[t] = data
    return(master)

# mscore calculation
def mscores(data,tick,factors,minVolume):
    mweights = np.array(factors)
    headers = data.columns[-len(factors):]

    # check for average volume threshold
    data = data[data['AvgVolume'] > minVolume]
    mdata = data[headers]

    mscores = np.dot(mdata,mweights)
    mbin = pd.DataFrame(data['Date']).set_index('Date')
    mbin[tick] = mscores
    return(mbin)

#%%
# retrieve list of tickers, set parameters
universe = pd.read_csv('companylist.csv')
marketcap = 250*10**6
firstTickers = directory(universe,marketcap)

# start and end date for sample
startDate = '2013-01-04'
endDate = '2016-10-28'

#%%
# download historical market data, can use either yahoo or quandl
# yahoo pulls more tickers, on average
datafile = 'yahoo'
quandlread = reader(datafile,startDate,endDate)
yahooread = reader(datafile,startDate,endDate)
s = time.time()
for t in firstTickers['Symbol']:
    #yahooread.data_dl(t,api='yahoo')
    #quandlread.data_dl(t,api='quandl')
    pass
print()
print('Time to Download :',time.time() - s)

#%%
# clean up ticker list
start = time.time()
droplist = []
for i in range(0,len(firstTickers)):
    filename = datafile+'/'+firstTickers['Symbol'][i]+'.csv'
    if os.path.exists(filename) == False:
        droplist.append(i)
tickers = firstTickers.drop(droplist)
tickers.reset_index(drop=True)
print('Time to clean ticker list :',time.time() - start)

#%%
# market values
nasd = pdr.DataReader('NDAQ','quandl',startDate,endDate)
nasd = nasd.reindex(index=nasd.index[::-1])
nasd['Returns'] = (nasd['AdjClose'] - nasd['AdjClose'].shift(1))/nasd['AdjClose'].shift(1)
nasd['Cumulative Returns'] = nasd['Returns'].cumsum()

#%%
# pull master data
start = time.time()
masterData = stockPanel(datafile,tickers['Symbol'])
print('Time to build master list :',time.time() - start)

#%%
start = time.time()
volumeThreshold = 5*10**4
mValues = pd.DataFrame()
retValues = pd.DataFrame()
factorWeights = [1,1,1,1,1,1,1,1]

for t in tickers['Symbol']:
    m = mscores(masterData[t],t,factorWeights,volumeThreshold)
    mValues = pd.concat([mValues,m],axis=1)
    retValues = pd.concat([retValues,masterData[t].set_index('Date')['Returns']],axis=1)

print('Time for Mscores and Returns :',time.time() - start)

#%%




