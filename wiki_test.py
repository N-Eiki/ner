from flair.data import Sentence
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.datasets import DataLoader, CONLL_03

import MeCab

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from predict import FLAIR_PRED

opt = argparse.ArgumentParser(description='args')
opt.add_argument('--data_path', default='/root/projects/data/NER_Flair')
opt.add_argument('--train_file', default='ja.wikipedia.conll')
opt.add_argument('--model_path', default='/root/projects/data/NER_Flair/models/sample/final-model.pt')

opt = opt.parse_args()


def main():
    columns = {0:'text',1:'ner'}
    corpus = ColumnCorpus(opt.data_path, columns, train_file=opt.train_file)
    test_data_loader = DataLoader(corpus.test, batch_size=1)
    tags= list()
    text = list()
   
    for i in tqdm(range(len(test_data_loader))):
        for sentence in (corpus.test[i].get_spans('ner')):
            tags.append(sentence.tag)
            text.append(sentence.text)

    df = pd.DataFrame(np.array([text, tags]).T)
    df.to_csv('wiki_test.csv',index=False)
if __name__=='__main__':
    main()