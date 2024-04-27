import random
import numpy as np
import string
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
 
from collections import Counter

"""nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')"""

class Skipgram_dataset:
    def __init__(self, data, labels, min_word_length: int):
        self.min_word_length = min_word_length
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # self.df = df
        preprocess_function = lambda x: self.normalize(
            x, 
            min_word_length=min_word_length,
            stop_words=self.stop_words
        )
        self.data = data.apply(preprocess_function) # df[data_column].apply(self.normalize)
        self.label = labels # df[labels_column]
        
        self.padding_token = '[pad]'
        self.count, self.word_to_idx, self.idx_to_word = self.lookup_table()
                
        self.n_gram_hash = {'1_gram': self.count}
        
        self.number_of_token = sum(self.count.values())
        self.subsampling()
        
        self.noise_dist = self.create_noise_dict()
        
    
    # Get class arguments
    def get_kwargs(self):
        return {
            'count': self.count,
            'word_to_idx': self.word_to_idx, 
            'idx_to_word': self.idx_to_word,
            'noise_dist': self.noise_dist
        }
    
    # Create dict mapping
    def lookup_table(self):
        tokens = [self.padding_token] + list(set(self.label))
        [tokens.extend(d) for d in self.data]
        
        # Word frequence
        count = Counter(tokens)
        
        # Mapping
        word_to_idx = {k:i for i, k in enumerate(count.keys())}
        idx_to_word = dict(zip(word_to_idx.values(), word_to_idx.keys()))        
        
        return count, word_to_idx, idx_to_word
    
    # Normalize text 
    @staticmethod
    def normalize(text:str, min_word_length:int=0, stop_words=[]):
        # Remove punctuation
        translator = str.maketrans('','',string.punctuation)
        
        tokens = []
        for i, token in enumerate(word_tokenize(text.lower())):
            # token = self.lemmatizer.lemmatize(token).translate(translator)
            token = token.translate(translator)
            if len(token) > min_word_length and token not in stop_words:
                tokens.append(token)
            
        return tokens
    
    # Create word distribution dict for getting word
    def create_noise_dict(self):
        alpha = 3/4
        noise_dist = {key: val ** alpha for key, val in self.count.items()}
        Z = sum(noise_dist.values())
        noise_dist_normalized = {key: val / Z for key, val in noise_dist.items()}
        return noise_dist_normalized
    
    # Create random K word that not in the context
    def negative_sample(self, K:int):
        return random.choices(
            list(self.noise_dist.keys()), 
            k=K, 
            weights=list(self.noise_dist.values())
        )
    
    # Get token whose probability is smaller than a threshold 
    def subsampling(self):        
        new_data = []
        for data in self.data:
            item = []
            for token in data:
                frac = self.count[token]/self.number_of_token
                threshold = 1e-5
                prob = 1 - np.sqrt(threshold / frac)
                # prob = (np.sqrt(frac/0.001)+1) * (0.001/frac)
                
                if np.random.sample() < prob:
                    item.append(token)
            new_data.append(item)
        
        self.data = new_data
        
    # Generate data
    def generate_skipgram_data(self, max_window_size:int, k:int):
        # Get indices of tokens in context
        def get_window_indices(i, max_len, window_size):
            if i > window_size: left = list(range(i-window_size, i))
            else: left = [-1]*(window_size-i) + list(range (0, i))
            if i+window_size < max_len: right = list(range(i+1, i+1+window_size))
            else: right = list(range(i+1, max_len)) + [max_len]*(window_size-(max_len-i-1))
            return left + right

        # Make skip gram of one size window
        skipgram = []
        for (data, label) in tqdm(zip(self.data, self.label), desc="Generate"):
            window_size = max_window_size
            data = data + [self.padding_token]
            
            # Create target word, context, non_context
            for idx in range(len(data)-1):
                pos_indices = get_window_indices(idx, len(data)-1, window_size)
                neg_indices = self.negative_sample(k)
                if data[idx] in self.word_to_idx:
                    input_idx = [self.word_to_idx[label]] # [self.word_to_idx[data[idx]]]
                    pos_idx = [self.word_to_idx[data[i]] for i in pos_indices] #  if data[i] in self.vocab
                    neg_idx = [self.word_to_idx[word] for word in neg_indices] #  if word in self.vocab
                    skipgram.append([input_idx, pos_idx, neg_idx])
                    
        return skipgram
    
    # Create vocab phrase ['New', 'York'] --> ['New_York']
    def create_n_gram(self, n_gram:int, min_delta:int, threshold:float):        
        key = f'{n_gram}_gram'
        if key in self.n_gram_hash:
            return
            
        final_n_gram_count = dict()
        
        n_gram_vocal = []
        for data in self.data:
            for i in range(len(data)-n_gram):
                word = data[i]
                next_word = data[i+1]
                    
                n_gram_vocal.append(word+'_'+next_word)

        n_gram_count = Counter(n_gram_vocal)
        for row_id, data in enumerate(tqdm(self.data, desc="N_gram")):
            new_tokens = []
            
            token_id = 0
            while token_id < len(data)-1:
                # Token
                word = data[token_id]
                next_word = data[token_id+1]
                
                # Frequence
                f_word = self.count[word]
                f_next_word = self.count[next_word]
                
                # N_gram
                n_gram_word = word+'_'+next_word
                f_n_gram = n_gram_count[n_gram_word]
                
                score = (f_n_gram - min_delta) / (f_word * f_next_word)
                
                if score > threshold:
                    token_id += 2
                    new_tokens.append(n_gram_word)
                    if n_gram_word not in final_n_gram_count:
                        final_n_gram_count[n_gram_word] = n_gram_count[n_gram_word]
                    self.count[word] -= 1
                    self.count[next_word] -= 1
                else:
                    new_tokens.append(word)
                    token_id += 1
            
            # Update
            self.data[row_id] == new_tokens
                  
        for new_vocab in final_n_gram_count.keys():
            self.count[new_vocab] = n_gram_count[new_vocab]
                    
        self.n_gram_hash[key] = final_n_gram_count
    
    def save(self, path, data=None):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'data': data}
        torch.save(ckpt, path)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        input_word = sample[0]
        pos_context = sample[1]
        neg_context = sample[2]
        
        return [
            torch.LongTensor(input_word),
            torch.LongTensor(pos_context),
            torch.LongTensor(neg_context)
        ]