from fastapi import FastAPI
from pydantic import BaseModel
import torch
import snscrape.modules.twitter as sntwitter
import pickle
from model.transformer import TransformerClassifier
from vocab.Vocab import Vocab
import torch.nn.functional as F
import feedparser
from urllib.parse import quote
import urllib.request


app = FastAPI()

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model=TransformerClassifier(len(vocab.word2idx),64,2,2,2,32)
model.load_state_dict(torch.load("sentiment_model.pt",map_location="cpu"))
model.eval()

label_map={0:"negative",1:"positive"}

class StockRequest(BaseModel):
    company:str
    max_tweets:int =10


def get_reddit_rss_posts(query, count=0):
    encoded_query = quote(query)
    rss_url = f"https://www.reddit.com/search.rss?q={encoded_query}"
    
    # Spoof User-Agent to mimic a real browser
    req = urllib.request.Request(
        rss_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        feed = feedparser.parse(response.read())

    return [entry.title for entry in feed.entries[:count]]


@app.post("/predict")
def predict(request:StockRequest):
    tweets=get_reddit_rss_posts(f"{request.company} stock",count=request.max_tweets)
    counts={"positive":0,"negative":0}

    for tweet in tweets:
        x=torch.tensor([vocab.encode(tweet,32)])
        with torch.no_grad():
            logits=model(x)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            sentiment = label_map[pred]
            counts[sentiment] += 1
    
    total=sum(counts.values())
    score = (counts["positive"] - counts["negative"]) / (counts["positive"] + counts["negative"] + 1e-5)

    if score > 0.2:
        recommendation = "BUY"
    elif score < -0.2:
        recommendation = "DON'T BUY"
    else:
        recommendation = "HOLD"

    return {
        "company": request.company,
        "tweet_count": total,
        "sentiment_counts": counts,
        "sentiment_score": round(score, 2),
        "recommendation": recommendation
    }
