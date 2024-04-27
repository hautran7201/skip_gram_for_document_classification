
# Document classification by skip gram (Negative sampling)

Perform embedding of words in the text so that it has the highest relationship with the embedding of the document label.


## Running



Create skip gram dataset for training
```python 
python data/generated_data/generate_data.py
```

Train skip gram model
```python 
python train_word2vec.py
```

Train classifier
```python 
python classifier.py
```


## Classification results


| Train | Test    | Validation |
| :---- | :------ | :--------- |
| `1.0` | `0.973` | `0.979`    |


## Reference

[Pythonic Excursions: Optimize Computational Efficiency of Skip-Gram with Negative Sampling](https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling#noise_dist)

