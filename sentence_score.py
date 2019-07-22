import math
import sys
#import tqdm
import torch

from data import Corpus
from data import TestCorpus
from utils import process_data


MODEL_FILE = sys.argv[1]
# CORPUS_PATH = '../language_model/data/swb-bpe-dp-extreme'
CORPUS_PATH = sys.argv[2]
VOCAB_PATH = sys.argv[3]
#################################################################
# Load data
#################################################################
corpus = TestCorpus(
    test_path=CORPUS_PATH,
    vocab_path=VOCAB_PATH,
    batch_size=1,
)

model = torch.load(MODEL_FILE, map_location='cpu')
model.cpu()

print('vocabulary size: ', len(corpus.vocab.idx2word))
print('sample words: ', corpus.vocab.idx2word[:10])


data_source = corpus.test
# Turn on evaluation mode which disables dropout.
model.eval()
#model.criterion.loss_type = 'nce'
#model.criterion.noise_ratio = 500
print('Rescoring using loss: {}'.format(model.criterion.loss_type))

# GRU does not support ce mode right now
eval_loss = 0
total_length = 0

scores = []

debug = False

with torch.no_grad():
    for data_batch in data_source:
        data, target, length = process_data(data_batch, cuda=False, sep_target=True)
#        loss, _ = model(data, length=length, target=target)
#        import pdb; pdb.set_trace()
 
#        loss = model.forward_fast(data, length=length, target=target)
        loss = model.forward_slow(data, length=length, target=target)
#        print (loss.tolist())
        print (loss.sum().item())
#        loss *= length.sum().item()
#        eval_loss += loss
#        total_length += length.sum().item()
#        score = - loss / math.log(2)  # change the base from e to 2
#        scores.append('{:.8f}'.format(score))
#
#print('PPL: ', math.exp(eval_loss / total_length))
#with open('./score.txt', 'w') as f_out:
#    f_out.write('\n'.join(scores))
