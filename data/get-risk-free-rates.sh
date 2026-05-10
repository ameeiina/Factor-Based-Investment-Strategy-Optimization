#! /usr/bin/env bash

# RISK-FREE RATES

# For the EU market - AAA 1-year government bond rates 
# 	ECB Data Portal_20240707200042.csv
# For the US market - 1-year T-Bill rates
#	US-T-BILLS.csv


#################
#
# EUROPEAN MARKET
#
#################

# 	Euro area yield curve for AAA rated countries published by ECB
#	https://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=165.YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y
#	Yield curve spot rate, 1-year maturity - Government bond, nominal, all issuers whose rating is triple A - Euro area (changing composition)
# 	download data from that website: 
EU_AAA_govBonds=ECB_Data_Portal_20240707200042.csv
DATEA="2015-01-01"    	#change to suit needs
DATEZ="2021-04-30"		#change to suit needs


cat $EU_AAA_govBonds | tr -d '"' | python -c "
import sys
import pandas as pd 

df = pd.read_csv(sys.stdin, skiprows=1, sep=',', names=['date', 'period', 'yield'], usecols=['date', 'yield'], index_col='date', parse_dates=['date'])
df.sort_index(inplace=True)
# select period: 
df = df.loc['$DATEA':'$DATEZ']
print('Avg EU risk-free rate for the period $DATEA--$DATEZ: {:.2f}%'.format(df.mean()[0]))
"

# Obtain the stoxx600 ETF tracker (EUR denominated, from iShares): EXSA.DE

yahoo EXSA.csv 2015-01-01 2020-12-31 EXSA.DE    # generates a local EXSA.csv

# VERY IMPORTANT!!! --> use EXSA.DE.csv, because latest EXSA.csv downloaded from Yahoo! Finance does not seem to have been adjusted for dividends
# Now you could use:
# $ ./annual.py 1 EXSA.DE.csv 'Adj Close' Rf=-0.6		// to see the sharpe ratio of this investment: 0.26




###########
#
# US MARKET
#
###########

# risk-free rate:  1.22%
# 	T-BILL 1-year rates, downloaded freely from https://www.quandl.com
# downloaded from that website and generated 'US-T-BILLS.csv'

wget https://www.quandl.com/api/v3/datasets/USTREASURY/YIELD.csv?api_key=mxvKxQGpdxyiEfiBEEwj -O US-T-BILLS.csv

head US-T-BILLS.csv
tail US-T-BILLS.csv

cat US-T-BILLS.csv | python -c "
import sys
import pandas as pd 

df = pd.read_csv(sys.stdin, header=0, usecols=['Date', '1 YR'], index_col=['Date'], parse_dates=['Date'])
df.sort_index(inplace=True)

# limit dates to 2015--2019Apr
df = df.loc['2015-01-01':'2019-04-30']
print('US risk-free rate: {:.2f}%'.format(df.mean()[0]))
" 


# Obtain 'SPY' ETF tracking the S&P500 from Yahoo! Finance

python -c "
import yfinance as yf
import datetime

SPY = yf.Ticker('SPY')

# for example, know the latest yield (dividends) for the SPY and highest-52week and lowest-52week:
SPY.info['shortName']		#SPDR S&P 500
SPY.info['quoteType']		#ETF
SPY.info['yield']			#0.014%
SPY.info['fiftyTwoWeekLow']	#272.99
SPY.info['fiftyTwoWeekHigh']#418.25

df = SPY.history(start=datetime.date(2015,1,1), end=datetime.date(2020,12,31))
df.to_csv('SPY.csv', header=True)
"

# Now you could use:
# $ ./annual.py 1 SPY.csv Close Rf=1.22		// to see the sharpe ratio of this investment: 1.14


