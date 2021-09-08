# coding: UTF-8
from tqdm import tqdm
import numpy
import numpy as np
import os
import time
import torch
import pickle
import argparse
#import joblib



from pytorch_pretrained import BertModel, BertTokenizer
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def build_bert_iterator(dataset, args):
    iter = bert_DatasetIterater(dataset, args.batch_size, args.device)
    return iter


class bert_DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, mask)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches



def load_dataset(path, pad_size=50):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for i, line in tqdm(enumerate(f)):
            lin = line.strip()
            # if not lin:
            #     print(i)
            #     continue
            token = tokenizer_english_cased.tokenize(lin)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = tokenizer_english_cased.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, seq_len, mask))
    return contents

def sent_process(model, data_iter, save_sent_file, is_cuda):
    print('Sentence processing begin...')
    with torch.no_grad():
        woz_embed, sent_embed = [], []
        state_id, end_id, accumulate = 0, 0, 0
        for i, (sent, len, mask) in enumerate(data_iter):
            if is_cuda:
                sent = sent.cuda()
                len = len.cuda()
                mask = mask.cuda()
            t_start = time.time()
            bert_outs = model(sent, attention_mask=mask, output_all_encoded_layers=False)
            t_end = time.time()
            print(t_end - t_start)
            # word_embed += bert_outs[0].tolist()
            # word_embed.append(bert_outs[0])
            sent_embed += bert_outs[1].tolist()


#            with open(save_words_file + "%d"%i, 'wb') as fw:
#                 pickle.dump(word_embed, fw)                
        with open(save_sent_file, 'ab+') as fs:
            pickle.dump(sent_embed, fs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='choose a model: Bert, ERNIE')
    parser.add_argument('--pretrain_model', type=str, default='bert',
                        help='choose a model: Bert, ERNIE')
    parser.add_argument('--english_cased_bert_path', type=str, default='./bert_pretrain/english_cased',
                        help='pretrain model path')
    parser.add_argument('--english_cased_bert_vocab_filename', type=str, default='bert-base-cased-vocab.txt',
                        help='pretrain model path,bert-base-cased-vocab.txt')
    parser.add_argument('--multilingual_cased_bert_path', type=str, default='./bert_pretrain/multilingual_cased/',
                        help='pretrain model path')
    parser.add_argument('--multilingual_cased_bert_vocab_filename', type=str, default='bert-base-multilingual-cased-vocab.txt',
                        help='pretrain model path')
    parser.add_argument('--data_path', type=str, default='./data/en-fr/',
                        help='data_path')
    parser.add_argument('--save_features_path', type=str, default='./data/en-fr/data_bert',
                        help='data_path')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='data_path')
    parser.add_argument('--tgt_lang', type=str, default='es',
                        help='data_path')
    parser.add_argument('--pad_size', type=int, default=50,
                        help='data_path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='data_path')

    args = parser.parse_args()



    tokenizer_english_cased = BertTokenizer.from_pretrained(os.path.join(args.english_cased_bert_path, args.english_cased_bert_vocab_filename))
    english_cased_bert_model = BertModel.from_pretrained(args.english_cased_bert_path)
    # tokenizer_multilingual_cased = BertTokenizer.from_pretrained(os.path.join(args.multilingual_cased_bert_path, args.multilingual_cased_bert_vocab_filename))
    # multilingual_cased_bert_model = BertModel.from_pretrained(args.multilingual_cased_bert_path)
    # tokenizer_vi_cased = BertTokenizer.from_pretrained(os.path.join(args.english_uncased_bert_path, args.english_uncased_bert_vocab_filename))


    is_cuda = torch.cuda.is_available()
    np.random.seed(1)
    torch.manual_seed(1)
    if is_cuda:
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        args.device = 'cuda'
        english_cased_bert_model = english_cased_bert_model.cuda()
        # multilingual_cased_bert_model = multilingual_cased_bert_model.cuda()
    print("Loading data and processing data...")
    print(repr(english_cased_bert_model) + "\n\n")
    # print(repr(multilingual_cased_bert_model) + "\n\n")
    train_src = load_dataset(os.path.join(args.data_path, 'train.'+ args.src_lang), args.pad_size)
    dev_src = load_dataset(os.path.join(args.data_path, 'valid.'+ args.src_lang), args.pad_size)
    test_src = load_dataset(os.path.join(args.data_path, 'test.'+ args.src_lang), args.pad_size)

    train_src_iter = build_bert_iterator(train_src, args)
    dev_src_iter = build_bert_iterator(dev_src, args)
    test_src_iter = build_bert_iterator(test_src, args)


    sent_process(english_cased_bert_model, train_src_iter, os.path.join(args.save_features_path, 'train_'+args.src_lang+'_sent.pkl'), is_cuda)
    print('train data process finished')
    sent_process(english_cased_bert_model, dev_src_iter, os.path.join(args.save_features_path, 'valid_'+args.src_lang+'_sent.pkl'), is_cuda)
    print('eval data process finished')
    sent_process(english_cased_bert_model, test_src_iter, os.path.join(args.save_features_path, 'test_' + args.src_lang + '_sent.pkl'), is_cuda)
    print("test data processing finished.")
    