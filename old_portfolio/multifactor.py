# Austin Griffith
# 12/5/2017
# Multifactor Trading
# Python 3.6.3

import pandas as pd
import numpy as np
import os
import datetime as dt
from pandas_datareader import data as pdr
import statsmodels.api as sm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import shutil

# fixes yahoo api for pandas datareader
import fix_yahoo_finance as yf
yf.pdr_override()

# creates data directory for yahoo data
if os.path.exists('data') == False:
    os.makedirs('data')

# reads into ticker directory (which also contains market cap data)
# limits tickers based on market cap criteria
# returns cleaned directory
class directory:
    def __init__(self,direc,cap):
        self.direc = direc
        self.cap = cap
        self.data = pd.DataFrame([])

    # cleans the ticker directory
    # looks for firms above a specified market cap
    def ticker_clean(self):
        data = self.direc

        for i in range(0,data.shape[0]):
            t = data['ticker'][i]
            if t.split('.')[1] == 'SH':
                new = t.split('.')[0]+'.SS'
                data['ticker'][i] = new

        data['mktshare'] = data['mktshare'].str.slice(0, -3)
        data['mktshare'] = data['mktshare'].str.replace(',', '')
        data['mktshare'] = pd.to_numeric(data['mktshare'])
        data = data[data['mktshare'] >= self.cap]
        data = data.reset_index(drop=True)
        self.data = self.data.append(data)
        return(data)

# uses directory to download files from yahoo finance api
# can be used to detemine correlation between mscore weights and returns
# calculates the values of the mscore using given weights
class reader:
    def __init__(self,tick,start='2011-01-04',end='2014-10-31'):
        self.tick = tick
        self.start = start
        self.end = end

    # downloads data from the patched yahoo api for a single ticker
    def data_dl(self):
        try:
            # import data from yahoo api, fixed
            data = pdr.get_data_yahoo(self.tick,self.start,self.end)

            # check if data has values, for troubleshooting
            if data.empty == True:
                print(' '+self.tick+' did not pull data')

            # added to prevent log errors
            const = 2.55*10**(-6)

            # Additional variables for financial data
            # simple returns
            # logarithmic returns
            # volatility
            # 5 day momentum
            # 20 day reversion
            # rolling 15 day average trade volume
            returns = data['Adj Close'] - data['Adj Close'].shift(1)
            returns = returns/data['Adj Close'].shift(1)
            log_ret = np.log(const + data['Adj Close'])
            log_ret = log_ret - np.log(const + data['Adj Close'].shift(1))
            volatility = np.log(const + log_ret.shift(1).rolling(window=20,center=False).std())
            momentum = np.log(const + data['Adj Close'].shift(1))
            momentum = momentum - np.log(const + data['Adj Close'].shift(5))
            reversion = np.log(const + data['Adj Close'].shift(20))
            reversion = reversion - np.log(const + data['Adj Close'].shift(1))
            avg_volume = data['Volume'].rolling(window=15,center=False).mean()

            # Append desired variables to dataframe
            data['Returns'] = returns
            data['Volatility'] = volatility
            data['Reversion'] = reversion
            data['Momentum'] = momentum
            data['Average Volume'] = avg_volume

            file = self.tick.split('.')[0]
            if os.path.exists('ss/'+file+'.ss_factor.csv') == True:
                factors = pd.read_csv('ss/'+file+'.ss_factor.csv')
                factors['Date'] = pd.to_datetime(factors['Date']).dt.date
                factors = factors.set_index('Date')
                data = data.merge(factors,left_index=True,right_index=True)
                if data.empty == False:
                    data.to_csv('data/'+self.tick+'.csv',sep=',')
                    print(' Data download successful: '+self.tick)
                else:
                    print(' Data download failed: '+self.tick+' -------> empty dataframe')

            elif os.path.exists('sz/'+file+'.sz_factor.csv') == True:
                factors = pd.read_csv('sz/'+file+'.sz_factor.csv')
                factors['Date'] = pd.to_datetime(factors['Date']).dt.date
                factors = factors.set_index('Date')
                data = data.merge(factors,left_index=True,right_index=True)
                if data.empty == False:
                    data.to_csv('data/'+self.tick+'.csv',sep=',')
                    print(' Data download successful: '+self.tick)
                else:
                    print(' Data download failed: '+self.tick+' -------> empty dataframe')

            else:
                print('.csv NOT FOUND')

        except Exception as e:
            print('Data download failed: '+self.tick+' -------> ',e)

    # solves for correlation between factors and returns
    # used in informing weights of M-score
    def m_weights(self):
        if os.path.exists('data/'+self.tick+'.csv') == True:
            data = pd.read_csv('data/'+self.tick+'.csv').set_index('Date')
            test = data.dropna(how='any')
            weights = test.corr(method='pearson', min_periods=1)
            values = weights[['Volatility','Momentum','Reversion','PB','PCF','PE','PS']].loc['Returns']
            norm = values/values.sum()
            norm = pd.DataFrame(norm).transpose().reset_index(drop=True)
            norm['ticker'] = self.tick
            norm = norm.set_index('ticker')

            # drop data if it has crazy large outliers
            if norm[abs(norm) < 5].dropna(how='any').empty == True:
                norm = pd.DataFrame([])
                print(self.tick+' correlation not useful')
        else:
            norm = pd.DataFrame([])
            print('.csv not found, correlation aborted')
        return(norm)

    # use m-weights to input m-scores into ticker .csvs
    # input should be series with proper labels
    def m_score(self,avg):
        if os.path.exists('data/'+self.tick+'.csv') == True:
            data = pd.read_csv('data/'+self.tick+'.csv')
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            data = data.set_index('Date')

            # mscore variables and calculations
            vol = avg['Volatility']*data['Volatility']
            rev = avg['Reversion']*data['Reversion']
            mom = avg['Momentum']*data['Momentum']
            pe = avg['PE']*data['PE']
            pcf = avg['PCF']*data['PCF']
            ps = avg['PS']*data['PS']
            pb = avg['PB']*data['PB']
            mscore = mom + rev + vol + pb + pcf + pe + ps
            data['Mscore'] = mscore

            # fit data to proper size of date range
            # any day with missing constants is
            const = -1*10**6
            dates = pd.bdate_range(self.start,self.end)
            dates = pd.DataFrame(dates)
            dates = pd.DataFrame(pd.to_datetime(dates[0]).dt.date)
            dates['merge'] = 0
            dates = dates.set_index(0)
            data = pd.concat([dates,data],axis=1).drop('merge',axis=1)
            data['Mscore'] = data['Mscore'].fillna(const)
            data = data.fillna(0)

            # removes days that consistently have empty values, namely market holidays
            year_start = int(start.split('-')[0])
            year_end = int(end.split('-')[0])
            day = ['0102','0103','0429','0430','0501','1001','1002','1003','1004','1005','1006','1007']

            for d in day:
                for y in range(year_start,year_end+1):
                    try:
                        i = str(y)+d
                        i = dt.datetime.strptime(i,'%Y%m%d').date()
                        location = data.index.get_loc(i)
                        data = data.drop(data.index[location])
                    except:
                        pass

            # for unique holidays in timer period
            holidays = pd.read_csv('holidays.csv')
            holidays = pd.to_datetime(holidays['Holidays'],format='%Y%m%d').dt.date
            for d in holidays:
                try:
                    location = data.index.get_loc(d)
                    data = data.drop(data.index[location])
                except:
                    pass

            data.to_csv('data/'+self.tick+'.csv',sep=',')
            print('M_score updated: '+self.tick)

# selects equities and relevant data for updating portfolio
class equity:
    def __init__(self,direc,iteration,frequency,value):
        self.direc = direc.reset_index(drop=True)
        self.iteration = iteration
        self.frequency = frequency
        self.value = value

    def select_top(self):
        i = self.iteration

        # negative to constant for missing tickers' mscore
        # makes sure mscore for that particular stock isn't chosen
        neg = -1*10**6

        # check for minimum average volume traded in past 15 days
        minimum_volume = 1*10**6

        mscore = pd.DataFrame(columns = ['Ticker','Close','Shares','Mscore'])
        mscore['Ticker'] = self.direc['ticker']

        # creates dataframe filled with tickers and respective mscores
        # if ticker breaks, values are input with lowest mscore
        # if average trading volume criteria isn't met, the mscore set to 0
        for k in range(0,self.direc.shape[0]):
            tick = self.direc['ticker'][k]
            try:
                if os.path.exists('data/'+tick+'.csv') == True:
                    data = pd.read_csv('data/'+tick+'.csv')
                    if data['Average Volume'][i] >= minimum_volume:
                        mscore['Mscore'][k] = data['Mscore'][i]
                        mscore['Close'][k] = data['Close'][i]
                        mscore['Shares'][k] = self.value/data['Close'][i]
                    else:
                        mscore['Mscore'][k] = neg
                        mscore['Close'][k] = 0
                        mscore['Shares'][k] = 0
                else:
                    mscore['Mscore'][k] = neg
                    mscore['Close'][k] = 0
                    mscore['Shares'][k] = 0
            except:
                mscore['Mscore'][k] = neg
                mscore['Close'][k] = 0
                mscore['Shares'][k] = 0

        # mscore dataframe catcher
        mscore = mscore.sort_values('Mscore',ascending=False)
        mscore = mscore.reset_index(drop=True)
        topm = mscore[0:100]

        # checks for broken selector
        # looks for neg value, which means values won't work
        check = topm.set_index('Mscore')
        try:
            if neg in check.index == True:
                topm = pd.DataFrame([])
        except:
            topm = pd.DataFrame([])
        return(topm)

# portfolio value calculation
class portfolio:
    def __init__(self,iteration,frequency,top):
        self.iteration = iteration
        self.frequency = frequency
        self.top = top.reset_index(drop=True)
        self.new_value = 0

    # calculates new portfolio list of top equities one period ahead
    def portfolio_value(self):
        i = self.iteration
        f = self.frequency

        new = 0
        for a in range(0,self.top.shape[0]):
            tick = self.top['Ticker'][a]
            if os.path.exists('data/'+tick+'.csv') == True:
                data = pd.read_csv('data/'+tick+'.csv')
                try:
                    if data['Close'] != 0:
                        new = new + data['Close'][i]*self.top['Shares'][a]
                    # if stock dissapears at new period, bump back to old value
                    else:
                        new = new + data['Close'][i-f]*self.top['Shares'][a]
                # if hit end of dataframe, read from previous value
                except:
                    new = new + data['Close'][i-f]*self.top['Shares'][a]
        self.new_value = new
        return(new)

# sell, buy and rebalance portfolio with the pass of each time period
class strategy:
    def __init__(self,direc,iteration,frequency,value):
        self.direc = direc
        self.iteration = iteration
        self.frequency = frequency
        self.value = value
        self.gain = 0
        self.lost = 0
        self.rebalance_lost = 0
        self.rebalance_gain = 0

    # buy the equities for the next time period porftolio
    # given a list of equities to buy using a difference in old and new top firm list
    def buy_equities(self,top,equities_to_buy):
        i = self.iteration
        f = self.frequency
        top = top.reset_index(drop=True)
        loss = 0

        # capture buy price of new equities to be bought
        for tick in equities_to_buy:
            data = pd.read_csv('data/'+tick+'.csv')
            try:
                current_price = data['Close'][i]
            # if reach end of dataframe, bump back to previous values
            except:
                current_price = data['Close'][i-f]
            shares = self.value/current_price
            loss = loss + value
            temp = [tick,current_price,shares]

            # add new values to equities list
            top.loc[len(top)] = temp

        self.lost = loss
        return(top)

    # sell equiteis using the difference between old and new top firms
    # outputs an updated top firm
    def sell_equities(self,top,equities_to_sell):
        i = self.iteration
        f = self.frequency
        top = top.reset_index(drop=True)
        profit = 0

        # capture sale price of each equity to be sold
        for k in range(0,top.shape[0]):
            tick = top['Ticker'][k]
            if tick in equities_to_sell:
                data = pd.read_csv('data/'+tick+'.csv')
                try:
                    current_price = data['Close'][i]
                    # if reach end of dataframe, bump back to previous values
                except:
                    current_price = data['Close'][i-f]
                profit = profit + (current_price*top['Shares'][k])

                # remove missing firms
                top = top[top['Ticker'] != tick]

        self.gain = profit
        return(top)

    # uses buy and sell values to rebalance the portfolio
    # creates the final update of the top firms dataframe
    def rebalance(self,top):
        top.reset_index(drop=True)
        reb_profit = 0
        reb_loss = 0

        # iterate throught the top firm dataframe
        for z in range(0,top.shape[0]):
            tick = top['Ticker'][z]
            old_shares = top['Shares'][z]
            current_price = top['Close'][z]
            new_shares = value/current_price
            diff_shares = new_shares - old_shares
            if diff_shares > 0:
                reb_loss = reb_loss + diff_shares*current_price
                temp = [tick,current_price,new_shares]
                top.loc[z] = temp
            if diff_shares < 0:
                reb_profit = reb_profit + diff_shares*current_price
                temp = [tick,current_price,new_shares]
                top.loc[z] = temp
            # if no values to update, pass
            else:
                pass

        self.rebalance_gain = reb_profit
        self.rebalance_lost = reb_loss
        return(top)

# performance metrics function
def performance(iteration,frequency,portfolio_values,profit_loss):
    # set up bin for final data
    final_data = []
    header = ['Time','Portfolio Values','Profit&Loss','Drawdowns','Returns']
    final_data.append(header)

    # get the largest portfolio value
    biggest_port = max(portfolio_values)

    for p in range(0,len(portfolio_values)):
        m = portfolio_values[p]
        # can't get required values in first loop
        if p == 0:
            returns = 0
        else:
            returns = (m - portfolio_values[p-1])
            returns = returns/portfolio_values[p-1]

        # calculate drawdown per period
        dd = (biggest_port - m)

        # set up row with desired variables, append to final data series
        row = [iteration,portfolio_values[p],profit_loss[p],dd,returns]
        final_data.append(row)
        iteration = iteration + frequency

    final_data = pd.DataFrame(final_data)
    final_data.columns = final_data.iloc[0]
    final_data = final_data.drop(final_data.index[0]).reset_index(drop=True)
    return(final_data)

# get market betas for each firm
def market_beta(ticker):
    # read market csv
    beta = pd.read_csv('beta_comparison.csv')
    beta['Date'] = pd.to_datetime(beta['Date']).dt.date
    beta = beta.set_index('Date')

    # beta bin
    modeled_beta = []
    header = ['Ticker','Beta']
    modeled_beta.append(header)

    # iterate through tickers, get market beta for each firm in the downloaded time period
    for tick in ticker['ticker']:
        if os.path.exists('data/'+tick+'.csv'):
            # pull firms and set index for merge
            data = pd.read_csv('data/'+tick+'.csv')
            data['Date'] = pd.to_datetime(data['Unnamed: 0'], format='%Y-%m-%d').dt.date
            data = data.drop('Unnamed: 0', axis=1)
            data = data.set_index('Date')

            # use beta and firm values to perform a linear regression
            # uses CAPM assumption in beta calculation
            # beta value is the coefficient of the linear approximation
            x = pd.DataFrame(data['Returns'])
            y = pd.DataFrame(beta['Returns'])
            y.columns.values[0] = 'Beta'
            merge = pd.merge(x, y, how='inner', left_index=True, right_index=True)
            merge.dropna(inplace=True)
            x = pd.DataFrame(merge['Returns'])
            y = pd.DataFrame(merge['Beta'])
            y = sm.tools.tools.add_constant(y)
            model = sm.OLS(x.astype(float),y.astype(float)).fit()
            row = [tick,model.params[1]]
            modeled_beta.append(row)
            print(tick+' market beta added')

    # edit modeled beta bin
    modeled_beta = pd.DataFrame(modeled_beta)
    modeled_beta.columns = modeled_beta.iloc[0]
    modeled_beta = modeled_beta.drop(modeled_beta.index[0]).reset_index(drop=True)
    return(modeled_beta)

##################### MAIN CODE #####################
# Initial Values
universe = pd.read_csv('ticker_universe.csv')
cap = 500*10**6
org = directory(universe,cap)
ticker = org.ticker_clean()

# start and end date for in-sample
start = '2011-01-04'
end = '2014-10-31'

# read in data and get m values for entire time period
mw = pd.DataFrame([])
for tick in ticker['ticker']:
    rd = reader(tick,start,end)
    rd.data_dl()
    # all this commented code was used for the correlation of weights
#     w = rd.m_weights()
#     if w.empty == False:
#         mw = mw.append(w)

# correlated values outputed into .csv
# mavg = mw.mean()
# m = pd.DataFrame([mavg])
# mw.to_csv('M-weights.csv',sep=',')
# m.to_csv('M-wavg.csv',sep=',')

m = pd.read_csv('M-wavg.csv')
mavg = m.iloc[0]

for tick in ticker['ticker']:
    rd = reader(tick,start,end)
    rd.m_score(mavg)



# ----------------------------------------------------
# portfolio value and number of firms
starting_port_value = 10*10**6
port_firms = 100

# starts at day 20
# updates portfolio every 20 days
frequency = 20
iteration = 22

# equity selection
value = starting_port_value/port_firms
eq = equity(ticker,iteration,frequency,value)
top_equities_list1 = eq.select_top()
top_equities_list1 = top_equities_list1.drop(['Mscore'],axis=1)
top_equities_list1 = top_equities_list1.reset_index(drop=True)

# list of portfolio values across the time period
portfolio_values = []
portfolio_values.append(starting_port_value)

# port values with iteration
port_values_final = []
header = ['Portfolio Value','Time']
port_values_final.append(header)
row = [starting_port_value,iteration]
port_values_final.append(row)

# iteration counter
counter = []
counter.append(iteration)

# transaction cost list
trans_cost = 0.001
transactions = []
transactions.append(starting_port_value*trans_cost)

# profit loss list
profit_loss = []
profit_loss.append(-starting_port_value-value*trans_cost)

# update portfolio name value for while loop
new_portfolio_value = starting_port_value
loop = 936
while iteration < loop:
    old_portfolio_value = new_portfolio_value
    iteration = iteration + frequency

    # reset list of top equities from previous period
    top_equities_list0 = top_equities_list1

    # new portfolio value and new top mscore firms
    port = portfolio(iteration,frequency,top_equities_list0)
    new_portfolio_value = port.portfolio_value()
    value = new_portfolio_value/port_firms
    eq = equity(ticker,iteration,frequency,value)
    new_top_firms = eq.select_top()

    # loop is broken
    if new_top_firms.empty == True:
        iteration = loop

    else:
        # -------------------------------------------
        # use strategy class to rebalance
        strat = strategy(ticker,iteration,frequency,value)

        # remove new from old to get equities to sell
        # remove old from new to get equities to buy
        old_top = set(top_equities_list0['Ticker'])
        new_top = set(new_top_firms['Ticker'])
        equities_to_sell = list(old_top - new_top)
        equities_to_buy = list(new_top - old_top)

        top_equities_list1 = strat.sell_equities(top_equities_list0,equities_to_sell)
        top_equities_list1 = strat.buy_equities(top_equities_list1,equities_to_buy)
        gain = abs(strat.gain)
        lost = abs(strat.lost)

        # rebalance portfolio list
        top_equities_list1 = strat.rebalance(top_equities_list1)
        rebalance_lost = abs(strat.rebalance_lost)
        rebalance_gain = abs(strat.rebalance_gain)
        top_equities_list1 = top_equities_list1.reset_index(drop=True)

        # -------------------------------------------
        # update lists
        # transaction costs
        tr = gain + lost + rebalance_lost + rebalance_gain
        tr = tr*trans_cost
        transactions.append(tr)

        # new portfolio values
        portfolio_values.append(new_portfolio_value)
        counter.append(iteration)
        row = [new_portfolio_value,iteration]
        port_values_final.append(row)

        # profit and loss
        pl = new_portfolio_value - old_portfolio_value - tr
        profit_loss.append(pl)
    print('iteration '+str(int((iteration-frequency)/frequency))+' is complete')

# final values
port_values_final = pd.DataFrame(port_values_final)
port_values_final.columns = port_values_final.iloc[0]
port_values_final = port_values_final.drop(port_values_final.index[0]).reset_index(drop=True)
final_data = performance(frequency,frequency,portfolio_values,profit_loss)
final_data.to_csv('In-Sample Data.csv')

# sharpe and average sharpe values
interest = 0.02
sharpe = (final_data['Portfolio Values'][final_data.shape[0]-1] - final_data['Portfolio Values'][0])/final_data['Portfolio Values'][0]
sharpe = (sharpe - interest)/final_data['Returns'].std()
avg_sharpe = (final_data['Returns'].mean() - interest)/final_data['Returns'].std()
dict_sh = {'Overall Sharpe':sharpe,'Averge Periodic Sharpe':avg_sharpe,'Std Ret':final_data['Returns'].std()}
sh = pd.DataFrame([dict_sh])
sh.to_csv('In-Sample Sharpe.csv')

# plot for portfolio values
final_data['Time'] = (final_data['Time'] - 20)/(252) + 2011
plt.figure(1)
init = final_data['Portfolio Values'][0]
plt.plot(final_data['Time'],final_data['Portfolio Values']/init,'k')
title = 'Portfolio Values for In-Sample'
plt.ylabel('Value ($ - Normalized)')
plt.xlabel('Time')
plt.title(title)
plt.savefig('In-Sample.pdf')
# reset figure for next plot
plt.clf()


# ----------------------------------------------------
# download market comparison

# market = '000300.SS'
# start = '2011-01-04'
# end = '2015-07-31'
# data = pdr.get_data_yahoo(market,start,end)
# returns = (data['Adj Close'] - data['Adj Close'].shift(1))
# returns = returns/data['Adj Close'].shift(1)
# data['Returns'] = returns
# data.to_csv('beta_comparison.csv',sep=',')

# get market betas for each ticker available in in-sample data
modeled_beta = market_beta(ticker)
modeled_beta = modeled_beta.set_index('Ticker')
modeled_beta.to_csv('Market Beta For Firms.csv')



# ----------------------------------------------------
# do out sample for 2015
# remove old data
if os.path.exists('data') == True:
    shutil.rmtree('data')
    os.makedirs('data')

# out-sample
start = '2014-11-01'
end = '2015-07-31'

# read in data and get m values for entire time period
mw = pd.DataFrame([])
for tick in ticker['ticker']:
    rd = reader(tick,start,end)
    rd.data_dl()

m = pd.read_csv('M-wavg.csv')
mavg = m.iloc[0]

for tick in ticker['ticker']:
    rd = reader(tick,start,end)
    rd.m_score(mavg)


# ----------------------------------------------------
# portfolio value and number of firms
starting_port_value = 10*10**6
port_firms = 100

# starts at day 20
# updates portfolio every 20 days
frequency = 20
iteration = 25

# equity selection
value = starting_port_value/port_firms
eq = equity(ticker,iteration,frequency,value)
top_equities_list1 = eq.select_top()
top_equities_list1 = top_equities_list1.drop(['Mscore'],axis=1)
top_equities_list1 = top_equities_list1.reset_index(drop=True)

# list of portfolio values across the time period
portfolio_values = []
portfolio_values.append(starting_port_value)

# port values with iteration
port_values_final = []
header = ['Portfolio Value','Time']
port_values_final.append(header)
row = [starting_port_value,iteration]
port_values_final.append(row)

# iteration counter
counter = []
counter.append(iteration)

# transaction cost list
trans_cost = 0.001
transactions = []
transactions.append(starting_port_value*trans_cost)

# profit loss list
profit_loss = []
profit_loss.append(-starting_port_value-value*trans_cost)

# update portfolio name value for while loop
new_portfolio_value = starting_port_value
loop = 190
while iteration < loop:
    old_portfolio_value = new_portfolio_value
    iteration = iteration + frequency

    # reset list of top equities from previous period
    top_equities_list0 = top_equities_list1

    # new portfolio value and new top mscore firms
    port = portfolio(iteration,frequency,top_equities_list0)
    new_portfolio_value = port.portfolio_value()
    value = new_portfolio_value/port_firms
    eq = equity(ticker,iteration,frequency,value)
    new_top_firms = eq.select_top()

    # loop is broken
    if new_top_firms.empty == True:
        iteration = loop

    else:
        # -------------------------------------------
        # use strategy class to rebalance
        strat = strategy(ticker,iteration,frequency,value)

        # remove new from old to get equities to sell
        # remove old from new to get equities to buy
        old_top = set(top_equities_list0['Ticker'])
        new_top = set(new_top_firms['Ticker'])
        equities_to_sell = list(old_top - new_top)
        equities_to_buy = list(new_top - old_top)

        top_equities_list1 = strat.sell_equities(top_equities_list0,equities_to_sell)
        top_equities_list1 = strat.buy_equities(top_equities_list1,equities_to_buy)
        gain = abs(strat.gain)
        lost = abs(strat.lost)

        # rebalance portfolio list
        top_equities_list1 = strat.rebalance(top_equities_list1)
        rebalance_lost = abs(strat.rebalance_lost)
        rebalance_gain = abs(strat.rebalance_gain)
        top_equities_list1 = top_equities_list1.reset_index(drop=True)

        # -------------------------------------------
        # update lists
        # transaction costs
        tr = gain + lost + rebalance_lost + rebalance_gain
        tr = tr*trans_cost
        transactions.append(tr)

        # new portfolio values
        portfolio_values.append(new_portfolio_value)
        counter.append(iteration)
        row = [new_portfolio_value,iteration]
        port_values_final.append(row)

        # profit and loss
        pl = new_portfolio_value - old_portfolio_value - tr
        profit_loss.append(pl)
    print('iteration '+str(int((iteration-frequency)/frequency))+' is complete')

# final values
port_values_final = pd.DataFrame(port_values_final)
port_values_final.columns = port_values_final.iloc[0]
port_values_final = port_values_final.drop(port_values_final.index[0]).reset_index(drop=True)
final_data = performance(frequency,frequency,portfolio_values,profit_loss)
final_data.to_csv('Out-Sample Data.csv')

# sharpe and average sharpe values
interest = 0.02
sharpe = (final_data['Portfolio Values'][final_data.shape[0]-1] - final_data['Portfolio Values'][0])/final_data['Portfolio Values'][0]
sharpe = (sharpe - interest)/final_data['Returns'].std()
avg_sharpe = (final_data['Returns'].mean() - interest)/final_data['Returns'].std()
dict_sh = {'Overall Sharpe':sharpe,'Averge Periodic Sharpe':avg_sharpe,'Std Ret':final_data['Returns'].std()}
sh = pd.DataFrame([dict_sh])
sh.to_csv('Out-Sample Sharpe.csv')

# plot for portfolio values
final_data['Time'] = (final_data['Time'] - 20)/(252) + 2014.9
plt.figure(1)
init = final_data['Portfolio Values'][0]
plt.plot(final_data['Time'],final_data['Portfolio Values']/init,'k')
title = 'Portfolio Values for Out-Sample'
plt.ylabel('Value ($ - Normalized)')
plt.xlabel('Time')
plt.title(title)
plt.savefig('Out-Sample.pdf')
# reset figure for next plot
plt.clf()



# ----------------------------------------------------
# example beta calculations, incomplete
# beta calculations to short
short_beta = []
header = ['Value','Returns']
short_beta.append(header)

# beta values
beta = pd.read_csv('beta_comparison.csv')
beta['Date'] = pd.to_datetime(beta['Date']).dt.date
beta = beta.set_index('Date')

for i in range(0,top_equities_list1.shape[0]):
    tick = top_equities_list1['Ticker'][i]
    cm = beta['Adj Close'][iteration]
    try:
        data = pd.read_csv('data/'+tick+'.csv')
        b = modeled_beta.loc[tick][0]
        s = top_equities_list1['Shares'][i]
        c = top_equities_list1['Close'][i]
        mkt = b*s*c
        ratio = mkt/starting_port_value
        beta_returns = data['Returns'] - ratio*beta['Returns'][iteration]
    except:
        mkt = 0
        beta_returns = 0
    short_beta.append([mkt,beta_returns])
short_beta = pd.DataFrame(short_beta)
short_beta.columns = short_beta.iloc[0]
short_beta = short_beta.drop(short_beta.index[0]).reset_index(drop=True)



