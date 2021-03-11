from flair.data import Sentence

from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.datasets import DataLoader, CONLL_03
import MeCab

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
opt = argparse.ArgumentParser(description='args')
opt.add_argument('--data_path', default='/root/projects/data/NER_Flair')
opt.add_argument('--train_file', default='ja.wikipedia.conll')
opt.add_argument('--model_path', default='/root/projects/data/NER_Flair/models/sample/final-model.pt')

opt = opt.parse_args()
class FLAIR_PRED:
    def __init__(self,):
        print('loading...')
        self.model = SequenceTagger.load(opt.model_path)
        self.tag = {}
        self.m = MeCab.Tagger('-Owakati')
        self.result = []

    def _parse_sentence(self, sentence):
        res = self.m.parse(sentence)
        return res

    def predict(self, sentence):
        sentence = self._parse_sentence(sentence)
        sentence = Sentence(sentence)
        self.model.predict(sentence)
        sentence.to_tagged_string()
        spans = sentence.get_spans('ner')

        for span in spans:
            self.result.append([span.text,span.tag])
        return self.result

    def get_tag(self,):
        return self.tag

    def reset(self):
        self.result=[]


if __name__=='__main__':
    model = FLAIR_PRED()
    columns = {0:'text',1:'ner'}
    corpus = ColumnCorpus(opt.data_path, columns, train_file=opt.train_file)
    test_data_loader = DataLoader(corpus.test, batch_size=1)
    result = pd.DataFrame()
    for i in tqdm(range(len(corpus.test))):
        t = corpus.test[i]#.get_spans('ner')
        # print(t)
        text = [token.text for token in t.tokens]
        text =''.join(text).replace(' ','')
        out=model.predict(text)
        tag = [o[1] for o in out]
        text= [o[0] for o in out]
        df1 = pd.DataFrame(np.array([tag,text])).T
        df1.columns = ['pred_tag','text']
        text = [s.text for s in corpus.test[0].get_spans('ner')]
        tags = [s.tag for s in corpus.test[0].get_spans('ner')]
        df2 = pd.DataFrame(np.array([tags,text])).T
        df2.columns = ['true_tag','text']
        df = pd.merge(df1, df2, on="text",how="left")
        if i==0:
            result = df
        else:
            result = pd.concat([result, df],axis=0)


        model.reset()

    result.to_csv('result.csv',index=False)
