from flair.data import Sentence
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.datasets import DataLoader, CONLL_03

import MeCab

import os
import argparse
from tqdm import tqdm
from predict import FLAIR_PRED

import numpy as np
import pandas as pd
opt = argparse.ArgumentParser(description='args')
opt.add_argument('--data_path', default='/root/projects/data/NER_Flair')
opt.add_argument('--train_file', default='ja.wikipedia.conll')
opt.add_argument('--model_path', default='/root/projects/data/NER_Flair/models/sample/final-model.pt')

opt = opt.parse_args()


def main():
    columns = {0:'text',1:'ner'}
    corpus = ColumnCorpus(opt.data_path, columns, train_file=opt.train_file)
    test_data_loader = DataLoader(corpus.test, batch_size=1)

    predictor = FLAIR_PRED()
    
    ##
    # result, score = predictor.model.evaluate(
    #     test_data_loader.dataset, out_path='eval.txt'
    # )
    # print(result.log_line)
    # print(score)
    # # 0.8857  0.8685  0.8770
    values=list()
    tags = list()
    string= ''
    for data in tqdm(test_data_loader.dataset):
        for d in  data:
            string+=d.text
        predictor.predict(string)
        for key, val in predictor.tag.items():
            for v in val:
                values.append(v)
                tags.append(key)


    df = pd.DataFrame(np.array([values, tags]).T)
    df.to_csv('wiki_test_pred.csv', index=False)

if __name__=='__main__':
    main()