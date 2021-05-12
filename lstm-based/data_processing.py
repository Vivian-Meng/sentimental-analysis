import torch 
import random
from torchtext.legacy import data, datasets

def generate_bigrams(x):
    n_grams = set(zip(*x[i:] for i in range (2)))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

SEED=1234

#seed is used to initialize the network
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic= True

# Field 主要是确定如何处理数据分词
#include_length():
# we use packed padded sentence, the padded is 0; and rnn needs to deal with non-padded, so we need to tell it the true length
TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm',preprocessing = generate_bigrams)


LABEL = data.LabelField(dtype=torch.float)

# torchtext.datasets 中是text-label 数据集 
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

#打印内容【text:"a","man"..., label: pos】
print(vars(train_data.examples[0]))

MAX_VOCAB_SIZE = 25000

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

# use word embeddings: glove.6b.100d
# initialize the pre-trained embeddings with Gaussian distributions, unk_init = torch.Tensor.normal_
TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d",unk_init=torch.Tensor.normal_)

# vocab_freqs returns [(word, occurrences),...]
print(TEXT.vocab.freqs.most_common(20))

# itos ['<unk','sd'...]
print(TEXT.vocab.itos[:10])


LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sort_within_batch = true, sort the batch with the sentence length
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)


# use bi-grams to fast training

