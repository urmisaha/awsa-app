'''
To run this file, two arguements are expected:
Command Example: python model_bilstm_trainable.py restaurant

Change the experimental setup values to match the experiment to be currently conducted

For any new domain, create folders inside dataset, models, ontology and logs folder
'''

import sys, os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import numpy as np
import random
import pickle, random, json, re, nltk

# experimental setup values
seed = 1234
batch_size = 1
sampling = "no"                                 # down|up|no  -  just for logs
weighted = "weighted"                           # unweighted/weighted - just for logs
dataamount = "50000"

print("Predicting AWSA")

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")

cn_words = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/embeddings/cn_nb.300_words.pkl', 'rb'))
cn_word2idx = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/embeddings/cn_nb.300_idx.pkl', 'rb'))
cn_embs = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/embeddings/cn_nb.300_embs.pkl', 'rb'))

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, scores, word2idx, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)
        tot_at = 0

        ''' Initialization of matrices '''
        scores_matrix = torch.ones((vocab_size, 1))
        weights_matrix = torch.ones((vocab_size, embedding_dim))
        for v in target_vocab:
            ''' initialize weights_matrix with conceptnet embeddings '''
            try:
                if v in ['_PAD','_UNK']:
                    weights_matrix[word2idx[v]] = torch.from_numpy(cn_embs[0])
                else:
                    weights_matrix[word2idx[v]] = torch.from_numpy(cn_embs[cn_word2idx[v]])
            except:
                pass
            ''' initialize scores_matrix with aspect scores '''
            if v in scores.keys():
                tot_at = tot_at + 1
                scores_matrix[word2idx[v], 0] = scores[v]

        # print("vocab_size = " + str(vocab_size) + " --- total aspect terms = " + str(tot_at))

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(weights_matrix)
        self.aspect_scores = nn.Embedding(vocab_size, 1)
        self.aspect_scores.weight = torch.nn.Parameter(scores_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)                                # 2 for bidirection, when dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):                                                       # x.shape =  torch.Size([10, 200]) (batch_size, seq_length)
        embeds = self.embedding(x)                                                      # embeds.shape =  torch.Size([10, 200, 100])

        ''' Multiplying the embeddings with the aspect scores from the aspect score layer '''
        scores = self.aspect_scores(x)                                                  # scores.shape =  torch.Size([10, 200, 1])
        scores1 = scores.repeat(1, 1, embedding_dim)                                    # scores1.shape =  torch.Size([10, 200, emb_dim])
        embeds = embeds*scores1
        
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

def onto_prediction(domain, test_sentence):
    # For Weighted Word Embeddings 
    with open("../../EMNLP2020/Aspect-weighted-SA/ontology/" + domain + "/scores.json", "r") as f:
        scores = json.load(f)

    word2idx = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/dataset/' + domain + '/word2idx.pkl', 'rb'))
    idx2word = pickle.load(open(f'../../EMNLP2020/Aspect-weighted-SA/dataset/' + domain + '/idx2word.pkl', 'rb'))
    
    vocab_size = len(word2idx) + 1

    target_vocab = word2idx.keys()

    model = BiRNN(scores, word2idx, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers).to(device)

    # Loading the best model
    filename = '../../EMNLP2020/Aspect-weighted-SA/models/' + domain + '/awsa_' + weighted + '_' + dataamount + '_' + str(seed) + '.pt'
    # print(filename)
    model.load_state_dict(torch.load(filename))

    test_sentence = re.sub('\d','0',test_sentence)
    test_sentence = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentence)
    test_sentence = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(test_sentence)]
    seq_len = 300
    features = np.zeros(seq_len, dtype=int)
    features[-len(test_sentence):] = np.array(test_sentence)[:seq_len]

    test_data = TensorDataset(torch.from_numpy(np.array(test_sentence)))
    # test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    h = model.init_hidden(batch_size)

    model.eval()

    total_labels = torch.LongTensor()
    total_preds = torch.LongTensor()

    # print(test_loader)
    # return 1
    # for inputs in test_loader:
    h = tuple([each.data for each in h])
    # inputs = torch.as_tensor(inputs).to(device)
    inputs = torch.as_tensor(test_sentence).to(device)
    print("inputs")
    print(inputs)
    inputs = inputs.unsqueeze(0)
    output = model(inputs, h)
    print("output")
    print(output)
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    pred = pred.to("cpu").data.numpy()

    print("Printing results::: ")
    print(pred)
    return pred