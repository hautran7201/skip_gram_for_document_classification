import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class Word2Vec(nn.Module):
    def __init__(self, word_to_idx, embedding_size):
        super(Word2Vec, self).__init__()
        self.vocab_size = len(word_to_idx)
        self.embedding_size = embedding_size
        self.word_to_idx = word_to_idx
        self.u_embeddings = nn.Embedding(self.vocab_size, embedding_size, sparse=True)   
        self.v_embeddings = nn.Embedding(self.vocab_size, embedding_size, sparse=True) 
        self.init_emb()
        
    def init_emb(self):
        initrange = 0.5 / self.embedding_size
        self.u_embeddings.weight.data.uniform_(-1, 1) # uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-1, 1) # uniform_(-0, 0)

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos).permute(0, 2, 1)
        
        # Positive 
        embed_v = self.v_embeddings(v_pos)
        # score  = torch.mul(embed_u, embed_v)
        score = torch.bmm(embed_v, embed_u)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        # Negative
        neg_embed_v = self.v_embeddings(v_neg)
        neg_score  = torch.bmm(neg_embed_v, embed_u)
        neg_score = torch.sum(neg_score, dim=1)        
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1*loss.mean() # .sum()/batch_size 
    
    def context_embedding(self, indices:list):        
        return self.v_embeddings(torch.tensor(indices))
    
    def save(self, path):
        ckpt = {
            'state_dict': self.state_dict(),
            'config': {
                'word_to_idx': self.word_to_idx,
                'embedding_size': torch.tensor(self.embedding_size)
            }
        }
        torch.save(ckpt, path)    