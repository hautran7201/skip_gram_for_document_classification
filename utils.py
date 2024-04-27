import torch
from data.generated_data.skipgram_dataset import Skipgram_dataset
from nltk.corpus import stopwords


def sentence_vectorize(corpus, model):
    result = []
    stop_words = set(stopwords.words('english'))
    for sentence in corpus:
        count = 0
        bucket = torch.zeros(model.embedding_size)
        for token in Skipgram_dataset.normalize(sentence, stop_words=stop_words):
            if token in model.word_to_idx:
                count += 1
                bucket += model.context_embedding(model.word_to_idx[token]).detach().cpu()
        bucket = bucket/count
        result.append(bucket)
        
    return result