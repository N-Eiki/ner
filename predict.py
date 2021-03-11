from flair.data import Sentence
from flair.models import SequenceTagger

import MeCab

import os
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
            if not span.tag in self.tag:
                self.tag[span.tag] =list()
            if span.text not in  self.tag[span.tag]:
                self.tag[span.tag].append(span.text)
        return self.tag

    def get_tag(self,):
        return self.tag

def main():
    predictor = FLAIR_PRED()
    
    while True:
        sentence = input('input>')
        if sentence=='q':
            break
        else:
            res = predictor.predict(sentence)
            print(res)

if __name__=='__main__':
    main()