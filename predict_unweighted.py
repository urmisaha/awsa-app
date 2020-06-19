'''
To run this file, two arguements are expected:
Command Example: python model_bilstm_trainable.py restaurant "the restaurant is good"
'''

import sys, os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import numpy as np
import pickle, random, json, re, nltk

# experimental setup values
seed = 1234
batch_size = 1
weighted = "weighted"                           # unweighted/weighted - just for logs
dataamount = "34000"

print("Predicting Unweighted")

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
device = torch.device("cpu")


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, word2idx, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)
  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)                                # 2 for bidirection, when dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):                                                       # x.shape =  torch.Size([10, 200]) (batch_size, seq_length)
        embeds = self.embedding(x)                                                      # embeds.shape =  torch.Size([10, 200, 100])

        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # h0.shape =  torch.Size([2, 10, 512]) : 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # c0.shape =  torch.Size([2, 10, 512])

        h1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # same as h0 
        c1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # same as c0

        # Forward propagate the weighted/unweighted embeddings to Bi-LSTM
        lstm_out, (hidden, cell) = self.lstm(embeds, (h0, c0))                          # lstm_out.shape =  torch.Size([10, 200, 1024]) (batch_size, seq_length, hidden_size*2) | hidden.shape =  torch.Size([2, 10, 512]) | cell.shape =  torch.Size([2, 10, 512])

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))     # after dropout: hidden.shape =  torch.Size([10, 1024])
        fc_out = self.fc(hidden)                                                        # when dropout - lstm_out.shape = ([10, 1])
        
        out = self.sigmoid(fc_out)
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device), weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden


output_size = 1
embedding_dim = 300
hidden_dim = 512
n_layers = 1

def unwt_prediction(domain, test_sentence):
    word2idx = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/dataset/' + domain + '/word2idx.pkl', 'rb'))
    idx2word = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/dataset/' + domain + '/idx2word.pkl', 'rb'))
    vocab_size = len(word2idx) + 1
    target_vocab = word2idx.keys()

    model = BiRNN(word2idx, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers).to(device)

    # Loading the best model
    filename = '../../EMNLP2020/Aspect-weighted-SA/models/' + domain + '/un' + weighted + '_' + dataamount + '_' + str(seed) + '.pt'
    model.load_state_dict(torch.load(filename))

    test_sentence = re.sub('\d','0',test_sentence)
    test_sentence = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentence)
    test_sentence = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(test_sentence)]
    seq_len = 300
    features = np.zeros(seq_len, dtype=int)
    features[-len(test_sentence):] = np.array(test_sentence)[:seq_len]

    h = model.init_hidden(batch_size)

    model.eval()

    total_labels = torch.LongTensor()
    total_preds = torch.LongTensor()

    h = tuple([each.data for each in h])
    inputs = torch.as_tensor(test_sentence).to(device)
    inputs = inputs.unsqueeze(0)
    output = model(inputs, h)
    print("Unweighted output")
    print(output)
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    pred = pred.to("cpu").data.numpy()

    print("Printing results::: ")
    print(pred)
    return pred
