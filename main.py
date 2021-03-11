from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import os
import argparse
opt = argparse.ArgumentParser(description='args')
opt.add_argument('--data_path', default='/root/projects/data/NER_Flair')
opt.add_argument('--train_file', default='ja.wikipedia.conll')
opt.add_argument('--model_path', default='/root/projects/data/NER_Flair/models/sample')


opt = opt.parse_args()
def main():
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)

    columns = {0:'text',1:'ner'}
    corpus = ColumnCorpus(opt.data_path, columns, train_file=opt.train_file)
    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    #分散表現の用意
    embedding_types = [
        FlairEmbeddings('ja-forward'),
        FlairEmbeddings('ja-backward')
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    #modelの定義
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True#CRFを使ってラベル間の依存関係を考慮する
        )
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(
        opt.model_path,
        learning_rate=0.1,
        mini_batch_size=128,
        max_epochs=150,
        )



if __name__ == '__main__':
    main()
    