import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="centered")
st.title("📈 Stock Sentiment Analyzer")

st.write("Enter a company name to analyze recent tweets and get a buy/hold/sell recommendation based on sentiment analysis.")

company = st.text_input("🔍 Company Name (e.g., Tesla, Apple, Microsoft)")
max_tweets = st.slider("Number of Tweets to Analyze", 5, 100, 10)

if st.button("Analyze Sentiment"):
    if not company.strip():
        st.warning("Please enter a company name.")
    else:
        with st.spinner("Fetching tweets and analyzing sentiment..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"company": company, "max_tweets": max_tweets}
                )
                result = response.json()
                print("Status Code:", response.status_code)
                print("Raw Response:", response.text)
                # Show results
                st.subheader(f"📊 Sentiment Analysis for: `{company}`")
                st.write(f"**Total Tweets Analyzed:** {result['tweet_count']}")
                st.write(f"**Sentiment Score:** `{result['sentiment_score']}`")
                st.write(f"**Recommendation:** 🟢 `{result['recommendation']}`" if result["recommendation"] == "BUY"
                         else f"**Recommendation:** 🔴 `{result['recommendation']}`" if result["recommendation"] == "DON'T BUY"
                         else f"**Recommendation:** 🟡 `{result['recommendation']}`")

                # Sentiment chart
                counts = result["sentiment_counts"]
                df = pd.DataFrame({"Sentiment": list(counts.keys()), "Count": list(counts.values())})
                fig = px.bar(df, x="Sentiment", y="Count", color="Sentiment", title="Sentiment Distribution")
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error: {e}")
