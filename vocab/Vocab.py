class Vocab:
    def __init__(self,texts,min_freq=1):
        self.word2idx={"<pad>":0,"<unk>":1}
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word]=len(self.word2idx)
    
    def encode(self,text,max_length):
        tokens=text.lower().split()
        ids=[self.word2idx.get(w,1) for w in tokens]
        return ids[:max_length]+[0]*(max_length-len(ids))