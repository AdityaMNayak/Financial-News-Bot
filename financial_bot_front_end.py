import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime
import time
from tvDatafeed import TvDatafeed, Interval
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path
import pickle

tv = TvDatafeed(chromedriver_path=None)
vader = SentimentIntensityAnalyzer()
st.set_page_config(initial_sidebar_state="expanded")
st.title('News-Fi')
st.markdown("""
    Providing stock analysis using real time news article sentiment analysis.
""")
st.set_option('deprecation.showPyplotGlobalUse', False)


def output(ticker, mag, txt):

    chartTitle = ticker
    data = tv.get_hist(ticker, 'NSE', interval=Interval.in_daily, n_bars=5000)
    placeholder1.pyplot(mpf.plot(data.tail(mag), type='candle', style='yahoo',
                                 title=chartTitle, tight_layout=True, figsize=(8, 3)))

    live_score = vader.polarity_scores(txt)
    st.markdown(f"Sentiment Score: {live_score['compound']}")

    model_file = Path(__file__).parents[1] / \
        'Financial-News-Bot/finalized_model.sav'
    loaded_model = pickle.load(open(model_file, 'rb'))
    live_score = pd.Series(live_score['compound'])
    returns = float(loaded_model.predict(
        live_score.values.reshape(-1, 1))) * 100
    returns = round(returns, 3)
    st.markdown(f"Predicted Next Day Returns: {returns}%")
    last_price = data.iloc[-1].close
    st.markdown(
        f"Predicted Next Day Returns in Rupees: {round(last_price*returns/100,2)}")


placeholder1 = st.video(
    "https://www.youtube.com/watch?v=73fnrywIhl8&ab_channel=IGUK")
placeholder2 = st.empty()


tickFile = Path(__file__).parents[1] / 'Financial-News-Bot/TickerList.csv'
df = pd.read_csv(tickFile)
df = df[['Symbol', 'Company Name']]


tick = st.sidebar.selectbox('Ticker:', df['Company Name'])
tick = df.loc[df['Company Name'] == tick, 'Symbol']
tick = tick.tolist()
tick = tick[0]


txt = st.sidebar.text_area("Sentiment Analyser Input :",
                           "Enter News article title, description or text")

start_button = st.sidebar.empty()
stop_button = st.sidebar.empty()


if start_button.button('start', key='start'):
    start_button.empty()
    if stop_button.button('stop', key='stop'):
        pass
    output(tick, 50, txt)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
