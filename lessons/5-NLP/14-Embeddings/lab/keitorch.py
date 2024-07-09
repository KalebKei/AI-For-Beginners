import builtins
import torch
import torchtext
import collections
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = None
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def load_dataset_from_csvs(dir='data/datasets', ngrams=1,min_freq=1):
    global vocab, tokenizer
    print("Loading dataset...")
    # Bug with torchtext.datasets 1.17.0. Error documented, but no fix, so I replicate. Yelp dataset already downloaded
    # train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(root='./data')

    train_dataset = pd.read_csv(os.path.join(dir, 'train.csv'))
    test_dataset = pd.read_csv(os.path.join(dir, 'test.csv'))

   
    train_dataset = train_dataset.values.tolist()
    test_dataset = test_dataset.values.tolist()
    
    classes = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    print('Building vocab...')
    counter = collections.Counter()

    for (label, line) in train_dataset:
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line),ngrams=ngrams))
    vocab = torchtext.vocab.vocab(counter, min_freq=min_freq)
    return train_dataset,test_dataset,classes,vocab