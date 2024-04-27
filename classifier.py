import torch
import os 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from model import Word2Vec
from utils import sentence_vectorize


def classifier(
        model_path,
        data_path,
        saving_path=None
    ):
    # Load Skipgram model
    config = torch.load(model_path)
    model = Word2Vec(**config['config'])
    model.load_state_dict(config['state_dict'])


    # Load data
    X_train = torch.load(os.path.join(data_path, 'X_train.pt'))
    y_train = torch.load(os.path.join(data_path, 'y_train.pt'))
    X_test = torch.load(os.path.join(data_path, 'X_test.pt'))
    y_test = torch.load(os.path.join(data_path, 'y_test.pt'))

    # Embedded data
    X_train_vec = sentence_vectorize(X_train, model)
    X_test_vec = sentence_vectorize(X_test, model)

    # ml model
    rf = RandomForestClassifier()
    rf.fit(X_train_vec, y_train)

    # cross validation
    cv_result = pd.DataFrame(
            cross_validate(rf, X_test_vec , y_test,scoring=['accuracy'], return_train_score=True, verbose=0, n_jobs=-1, cv=6)
        ).rename(columns={'test_accuracy':'val_accuracy'}).iloc[:,2:]

    # score
    val_score = cv_result['val_accuracy'].mean()
    train_score = cv_result['train_accuracy'].mean() 
    test_score = accuracy_score(y_true=y_test,y_pred=rf.predict(X_test_vec))

    if saving_path:
        joblib.dump(rf, saving_path)


    print('Training Score: ',train_score)
    print('Validation Score: ',val_score) 
    print('Test Score: ',test_score)


if __name__ == '__main__':
    classifier(
        model_path = 'checkpoint/word2vec/model.pt',
        data_path = 'data/bbc_data',
        saving_path = 'checkpoint/classifier/model.joblib'
    )