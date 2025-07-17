import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from model.transformer import TransformerClassifier
from vocab.Vocab import Vocab
from sklearn.model_selection import train_test_split
import pickle

    
class SentimentDataset(Dataset):
    def __init__(self,df,vocab,max_length=32):
        self.texts=df['text'].tolist()
        self.labels=df['label'].astype(int).map({-1:0,1:1}).tolist()
        self.vocab=vocab
        self.max_length=max_length
    
    def  __getitem__(self,idx):
        x=torch.tensor(self.vocab.encode(self.texts[idx],self.max_length))
        y=torch.tensor(self.labels[idx])
        return x,y
    
    def __len__(self):
        return len(self.labels)
    
df=pd.read_csv("data/sentiment.csv")
train_df,val_df=train_test_split(df,test_size=0.2)
vocab=Vocab(df['text'].tolist())

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
    
train_ds=SentimentDataset(train_df,vocab)
train_loader=DataLoader(train_ds,batch_size=32,shuffle=True)

model=TransformerClassifier(
    vocab_size=len(vocab.word2idx),
    embed_dim=64,
    num_heads=2,
    num_classes=2,
    num_layers=2,
    max_length=32
)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)


for epoch in range(20):
    model.train()
    for x,y in train_loader:
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} : loss {loss.item():.4f}")

torch.save(model.state_dict(),"sentiment_model.pt")