import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from data.generated_data.skipgram_dataset import CustomDataset
from model import Word2Vec 


def train(
        batch_size = 512,
        embedding_size = 300,
        learning_rate = 0.01,
        epochs = 10,
        device = 'cpu',
        saving_path = 'checkpoint/model.pt',
        data_path = 'data/generated_data/train_data/data.pt'
    ):


    # Data 
    dataset = torch.load(data_path)
    word_to_idx = dataset['kwargs']['word_to_idx']
    train_data = dataset['data']
    dataset = CustomDataset(train_data)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Model
    model = Word2Vec(word_to_idx, embedding_size=embedding_size)
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
            

    model.save(saving_path)


if __name__ == "__main__":
    # Parameters
    batch_size = 512
    embedding_size = 300
    learning_rate = 0.01
    epochs = 10
    device = 'cpu'
    saving_path = 'checkpoint/word2vec/model.pt'
    data_path = 'data/generated_data/train_data/data.pt'

    train(
        batch_size,
        embedding_size,
        learning_rate,
        epochs,
        device,
        saving_path,
        data_path
    )