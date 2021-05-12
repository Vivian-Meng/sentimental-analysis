import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import TEXT


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

#dropout used to drop out some neurons in the process to avoid overfit

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        # pad_index to tell the embedding not to deal with the pad tokens <pad>
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        #self.rnn = nn.RNN(embedding_dim, hidden_dim)

        # use lstm instead of rnn, output of lstm is output, final hidden state and cell state
        # bidirectional, num_layers achieveds the biderictionality and multiple layers
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        # lstm final hidden states has a forward and backward component, will be concatenated together
        # thus the size is 2 
        #linear layer fc takes the  takes the final hidden state and feeds it through a fully 
        # connected layer, $f(h_T)$, transforming it to the correct output dimension.
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        #can only drop out intermediate layers
        self.dropout = nn.Dropout(dropout)
        
    # pass the lengths of the text to deal with the packed padded sequence
    def forward(self, text,text_lengths):

        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        
        #output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        # pack senquence, will make rnn only process non-padded one
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        #output also hidden, cell which are tensors, using packed sequence made both of them non 0
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence, transform from packed sequence to tensor 
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # final hidden state, shape [num layers * num directions, batch size, hid dim]
        #  ordered [forward_layer_0, backward_layer_0,...forward_layer_n, backward_layer_n]
        #  we want the final forward hidden state and the backward one, so we have to get [-1,;,;],[-2,;,;]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden)


model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)
##count the number of the trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

# we need to initialze the embedding layer with pre-trained embedding
model.embedding.weight.data.copy_(pretrained_embeddings)

# we need to initialize the unkown and pad tokens into 0 

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)