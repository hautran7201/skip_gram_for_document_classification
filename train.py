import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from skipgram_dataset import CustomDataset
from model import Word2Vec 

# Parameters
batch_size = 512
embedding_size = 300
learning_rate = 0.01
epochs = 20
device = 'cpu'


# Data 
path = ''
dataset = torch.load()
word_to_idx = dataset['kwargs']['word_to_idx']
train_data = dataset['data']
dataset = CustomDataset(train_data)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model
model = Word2Vec(len(word_to_idx), embedding_dim=embedding_size)
optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)


# Train 
training_loss = []
model.train()
for epoch in range(epochs):
    
    pbar = tqdm(train_dataloader)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
                
        input_word, pos_context, neg_context = batch
        input_word = input_word.to(device)
        pos_context = pos_context.to(device)
        neg_context = neg_context.to(device)
        
        loss = model(input_word, pos_context, neg_context, batch_size)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            training_loss.append(loss)
            
        if i % 10 == 0:
            pbar.set_description(
                f"Epoch: {epoch}"
                + f" Loss: {loss}"
            )