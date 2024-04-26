import pandas as pd
from sklearn.model_selection import train_test_split 
from skipgram_dataset import Skipgram_dataset

# Load data
path = 'data/bbc_data/bbc_data.csv'
df = pd.read_csv(path)

# Data
X = df['data']
Y = df['labels']
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    Y, 
    test_size=0.20, 
    random_state=2024, 
    stratify=Y,
    shuffle=True
)

skg_dataset = Skipgram_dataset(    
    data=X_train, 
    labels=y_train,
    min_word_length=0
)
print('[1 Gram]')
print('Number of tokens:', skg_dataset.number_of_token)
print('Number of vocab:', len(skg_dataset.word_to_idx))


n_gram=2
skg_dataset.create_n_gram(n_gram=n_gram, min_delta=5, threshold=0.01)
print('\n==================================')
print('[1 Gram, 2 Gram]')
print('Number of tokens:', skg_dataset.number_of_token)
print('Number of vocab:', len(skg_dataset.word_to_idx))


k = 10 # Number of negative words (Words not in the context)
window_size = 6 # Number of context words = window size*2
train_data = skg_dataset.generate_skipgram_data(window_size, k)
print('Number of sample:', len(train_data))


# Save data
save_path = 'data/generated_data/data.pt'
skg_dataset.save(path=save_path, data=train_data)