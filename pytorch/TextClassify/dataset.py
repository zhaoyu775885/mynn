# -*- coding: utf-8 -*-

from torchtext import data

class Sentiment():
    def __init__(self, data_path, 
                 init_token='<SOS>', 
                 eos_token='<EOS>'):
        self.data_path = data_path
        TEXT = data.Field(sequential=True,
                          init_token=init_token, 
                          eos_token=eos_token, 
                          lower=True,
                          use_vocab=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        
        self.dataset = data.TabularDataset(self.data_path, 'TSV', 
                                           skip_header=True, 
                                           fields=[('PhraseId', LABEL), 
                                                   ('SentenceId', LABEL),
                                                   ('Phrase', TEXT), 
                                                   ('Sentiment', LABEL)])
        TEXT.build_vocab(self.dataset)
        self.vocab = TEXT.vocab
        self.vocab_size = len(self.vocab)
        self.iterator = None
        
    def build_iterator(self, batch_size, train=True):
        if self.iterator == None:
            if train:
                self.iterator = data.BucketIterator(
                    self.dataset, 
                    batch_size=batch_size,
                    train=train,
                    sort=True,
                    sort_key=lambda x: len(x.Phrase),
                    repeat=False)
            else:
                self.iterator = data.Iterator(
                    self.dataset,
                    batch_size=batch_size,
                    train=train,
                    sort=False,
                    repeat=False)
        return self.iterator
    
    def raw(self, ID_list):
        raw_list = []
        length, bsz = ID_list.shape
        for _ in range(bsz):
            tmp_list = []
            for i in range(length):
                tmp_list.append(self.vocab.itos[ID_list[i,_]])
            tmp = ' '.join(tmp_list)
            raw_list.append(tmp)
        return raw_list

    
if __name__ == '__main__':
    data_path = '/home/zhaoyu/Datasets/NLPBasics/sentiment/train.tsv'
    dataset = Sentiment(data_path)
    iterator = dataset.build_iterator(1, False)
    print(iterator.train, iterator.sort, iterator.shuffle)
    for i, batch in enumerate(iterator):
        print(i, batch.Sentiment, dataset.raw(batch.Phrase))
        if i+1==3:
            break
        
