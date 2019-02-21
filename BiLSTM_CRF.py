import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from numpy import random as rd

torch.manual_seed(1)
rd.seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNSEEN_WORD = "<UNSEEN>"

def prepare_seq(seq,_2ix):
    return torch.LongTensor([_2ix[element] if element in _2ix else _2ix[UNSEEN_WORD] for element in seq])

class BiLSTM_CRF(nn.Module):
    
    def __init__(self,vocab_sz,embed_dim,hid_dim,tag2ix):
        super(BiLSTM_CRF,self).__init__()
        self.hid_dim = hid_dim
        self.tag2ix = tag2ix
        self.embeds = nn.Embedding(vocab_sz,embed_dim)
        self.lstm = nn.LSTM(embed_dim,hid_dim//2,num_layers=1,bidirectional=True)
        self.hid2tag = nn.Linear(hid_dim,len(tag2ix))
        self.transition = nn.Parameter(torch.randn(len(tag2ix),len(tag2ix)))
        self.transition.data[tag2ix[START_TAG],:] = -10000
        self.transition.data[:,tag2ix[STOP_TAG]] = -10000
        
    def _init_hidden(self):
        return torch.randn((2,1,self.hid_dim//2)),torch.randn((2,1,self.hid_dim//2))
        
    def _get_lstm_features(self,sent):
        embeds = self.embeds(sent).view(len(sent),1,-1)
        hidden = self._init_hidden()
        hiddens,hidden = self.lstm(embeds,hidden)
        feats = self.hid2tag(hiddens)
        return feats
        
    def _get_log_prob(self,feats,tags):
        ret = torch.zeros(1)
        tags = torch.cat((torch.LongTensor([self.tag2ix[START_TAG]]),tags))
        for i,feat in enumerate(feats):
            ret += feat[tags[i+1]] + self.transition[tags[i+1]][tags[i]]
        return ret + self.transition[self.tag2ix[STOP_TAG]][tags[-1]]
    
    def _get_log_partition_function(self,feats):
        forward_var = torch.full((len(self.tag2ix),), -10000.)
        forward_var[self.tag2ix[START_TAG]] = 0.
        for feat in feats:
            forward_var = torch.logsumexp(forward_var.view((1,-1))+feat.view((-1,1))+self.transition,dim=1)
        terminal_var = forward_var+self.transition[self.tag2ix[STOP_TAG],:]
        return torch.logsumexp(terminal_var,dim=0)

    def loss(self,sent,tags):
        feats = self._get_lstm_features(sent).view((len(sent),-1))
        log_prob = self._get_log_prob(feats,tags)
        log_part = self._get_log_partition_function(feats)
        return log_part - log_prob

    def forward(self,sentence):
        feats = self._get_lstm_features(sentence)
        forward_var = torch.full((len(self.tag2ix),),-10000.)
        forward_var[self.tag2ix[START_TAG]] = 0.
        backwards = []
        #iteration
        for feat in feats:
            temp = forward_var.view((1,-1)) + feat.view((-1,1)) + self.transition
            forward_var = temp.max(dim=1)[0]
            backwards = [temp.max(dim=1)[1]] + backwards
        forward_var += self.transition[:,self.tag2ix[STOP_TAG]]
        id_ = forward_var.argmax().item()
        #recurrence
        best_path_id = [id_]
        for backward in backwards:
            id_ = backward[id_].item()
            best_path_id = [id_]+ best_path_id
        assert id_ == self.tag2ix[START_TAG]
        return best_path_id[1:]