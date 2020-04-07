from torchtext import data
from torchtext import datasets
import os

class UDPOS():
    def __init__(self, data_dir,
                 init_token='<SOS>',
                 eos_token='<EOS>'):
        self.data_dir = data_dir
        # file_path = lambda x, y: os.path.join(x, [e for e in os.listdir(x) if y in e][0])
        # self.train_path = file_path(self.data_dir, 'train')
        # self.valid_path = file_path(self.data_dir, 'dev')
        # self.test_path = file_path(self.data_dir, 'test')
        TEXT = data.Field(sequential=True,
                          init_token=init_token,
                          eos_token=eos_token,
                          lower=True,
                          use_vocab=True)
        LABEL_COARSE = data.Field(sequential=True, pad_token=None, unk_token=None, use_vocab=True)
        LABEL_FINE = data.Field(sequential=True, pad_token=None, unk_token=None, use_vocab=True)

        self.train, self.valid, self.test = datasets.UDPOS.splits(fields=[('text', TEXT),
                                                                          ('label_coarse', LABEL_COARSE),
                                                                          ('label_fine', LABEL_FINE)],
                                                                  root=self.data_dir)
        TEXT.build_vocab(self.train)
        LABEL_COARSE.build_vocab(self.train)
        LABEL_FINE.build_vocab(self.train)
        self.vocab = TEXT.vocab
        self.label_vocab = LABEL_COARSE.vocab

    def build_iterator(self, batch_size, train=True):
        self.test_iterator = data.BucketIterator(
            self.train,
            batch_size=1,
            train=False,
            sort=False,
            shuffle=False,
            repeat=False)
        return self.test_iterator

    def raw(self, ID_list):
        raw_list = []
        length, bsz = ID_list.shape
        for _ in range(bsz):
            tmp_list = []
            for i in range(length):
                tmp_list.append(self.vocab.itos[ID_list[i, _]])
            tmp = ' '.join(tmp_list)
            raw_list.append(tmp)
        return raw_list

if __name__ == '__main__':
    data_dir = '/home/zhaoyu/Datasets/NLPBasics/SeqLabel/'
    dataset = UDPOS(data_dir)
    iterator = dataset.build_iterator(1, False)
    for i, batch in enumerate(iterator):
        print(i, batch.text, dataset.raw(batch.text), '\n', batch.label_coarse)
        if i == 0:
            break