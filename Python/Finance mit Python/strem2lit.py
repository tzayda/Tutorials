import streamlit as st
import pandas as pd 
import pandas_datareader as web
import numpy as np 
from datetime import datetime, date
from scipy.optimize import minimize

st.write("""
# Mathematischer Portfolio Optimierer
""")

symbol = np.array(pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/0.csv")["Symbol"].values)

options = st.multiselect(
    "Ticker Namen",
    list(symbol),
    ["AAPL", "FB"]
)

start_date = st.date_input("Start Datum", value = datetime.date(datetime(2018,1,1)))
end_date = st.date_input("End Datum", value = datetime.today())

df = pd.DataFrame()

for o in options:
    df[o] = web.DataReader(o, data_source= "yahoo", start=start_date, end=end_date)["Adj Close"]

st.header("Aktien Kurs")
st.line_chart(df)

st.header("Daten Statistik")
st.write(df.describe())

st.header("Korrelation der Aktien")
st.write(df.pct_change().corr())

log_ret = np.log(df/df.shift(1))
  
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean()*weights)*252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252,weights)))
    sr = ret/vol

    return np.array([ret, vol, sr])

def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2]* -1

def check_sum(weights):
    return np.sum(weights) - 1

cons = ({"type" : "eq", "fun" : check_sum})

init_guess = [1/len(options) for i in options]

zero_one = ((0, 1), )
bounds = ()

for i in range(0, len(options)):
    bounds += zero_one

opt_result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=cons)

st.header("Optimale Portfolio Gewichtung")
for i in range(len(options)):
    st.write(options[i], opt_result.x[i])

data = get_ret_vol_sr(opt_result.x)

st.header("Optimales Portfolio")
st.write("Rendite", data[0])
st.write("Volatilität", data[1])
st.write("Sharpe Ratio", data[2])

