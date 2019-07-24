import math
import sys
#import tqdm
import torch

from data import Corpus
from data import TestCorpus
from utils import process_data

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('MODEL_FILE',  type=str)
parser.add_argument('CORPUS_PATH', type=str)
parser.add_argument('VOCAB_PATH',  type=str)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--offset', dest='offset', type=float, default=0.0)

args = parser.parse_args()
#################################################################
# Load data
#################################################################
corpus = TestCorpus(
    test_path=args.CORPUS_PATH,
    vocab_path=args.VOCAB_PATH,
    batch_size=1,
)

model = torch.load(args.MODEL_FILE, map_location='cpu')
model.cpu()

#print('vocabulary size: ', len(corpus.vocab.idx2word))
#print('sample words: ', corpus.vocab.idx2word[:10])

data_source = corpus.test
model.eval()

eval_loss = 0
total_length = 0

scores = []

debug = False

with torch.no_grad():
    for data_batch in data_source:
        data, target, length = process_data(data_batch, cuda=False, sep_target=True)
        if args.normalize: 
            loss = model.forward_slow(data, length=length, target=target)
        else:
            loss = model.forward_fast(data, length=length, target=target)

        if args.offset != 0:
            loss += length.item() * args.offset
        print (loss.sum().item())
