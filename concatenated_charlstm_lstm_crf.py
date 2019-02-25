from preprocessing import *

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

#preprocessing
train_data = preprocessing(TRAIN_W_PATH)
test_data = preprocessing(TEST_W_PATH)

UNSEEN_WORD = "<UNSEEN_WORD>"
UNSEEN_CHAR = "<UNSEEN_CHAR>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"

word2ix = {}
char2ix = {}
tag2ix = BMES2ix
for sentence in train_data[0]:
    for word in sentence:
        if not word in word2ix:
            word2ix[word] = len(word2ix)
        for char in word:
            if not char in char2ix:
                char2ix[char] = len(char2ix)
word2ix[UNSEEN_WORD] = len(word2ix)
char2ix[UNSEEN_CHAR] = len(char2ix)
tag2ix[START_TAG] = len(tag2ix)
tag2ix[STOP_TAG] = len(tag2ix)

ix2word = {ix:word for word,ix in word2ix.items()}
ix2char = {ix:char for char,ix in char2ix.items()}
ix2tag = {ix:tag for tag,ix in tag2ix.items()}

VOCAB_SZ = len(word2ix)
CHARSET_SZ = len(char2ix)

def process_tag(tags):
    return torch.LongTensor([tag2ix[ele] for ele in tags])

def process_sent(sent):
    words = torch.LongTensor([word2ix[word] if word in word2ix else word2ix[UNSEEN_WORD] for word in sent])
    chars = torch.LongTensor([char2ix[char] if char in char2ix else char2ix[UNSEEN_CHAR] for char in "".join(sent)])
    ix = 0
    ix_seq = []
    for word in sent:
        ix_seq.append(ix)
        ix += len(word)
    ix_seq.append(ix)
    return words,chars,ix_seq


train_data = [(process_sent(sent), process_tag(tags)) for sent,tags in zip(train_data[0],train_data[1])]

test_data = [(process_sent(sent), process_tag(tags)) for sent,tags in zip(test_data[0],test_data[1])]


class ConcatCharLSTM_LSTM_CRF(nn.Module):

    def __init__(self,vocab_sz,word_embed_dim,charset_sz,char_embed_dim,char_hid_dim,hid_dim,tag2ix):
        super(ConcatCharLSTM_LSTM_CRF,self).__init__()
        self.tag2ix,self.tagset_sz = tag2ix,len(tag2ix)
        self.char_hid_dim,self.hid_dim = char_hid_dim,hid_dim
        self.word_embed = nn.Embedding(vocab_sz,word_embed_dim)
        self.char_embed = nn.Embedding(charset_sz,char_embed_dim)
        self.char_lstm = nn.LSTM(char_embed_dim,char_hid_dim//4,bidirectional=True)
        self.lstm = nn.LSTM(word_embed_dim+char_hid_dim,hid_dim//2,bidirectional=True)
        self.hid2tag = nn.Linear(hid_dim,self.tagset_sz)
        self.transition = nn.Parameter(torch.randn((self.tagset_sz,self.tagset_sz)))
        self.transition.data[self.tag2ix[START_TAG],:] = -10000
        self.transition.data[:,self.tag2ix[STOP_TAG]] = -10000

    def _init_char_hidden(self):
        return torch.randn((2,1,self.char_hid_dim//4)), torch.randn((2,1,self.char_hid_dim//4))

    def _init_hidden(self):
        return torch.randn((2,1,self.hid_dim//2)), torch.randn((2,1,self.hid_dim//2))

    def _get_lstm_features(self,sent):
        words,chars,ix_seq = sent
        char_hids = self.char_lstm(self.char_embed(chars).view((len(chars),1,-1)),self._init_char_hidden())[0].view((len(chars),1,-1))
        word_embeds = self.word_embed(words).view((len(words),1,-1))
        ##TODO:generate inputs using char_hids, word_embeds and ix_seqs
        char_feats = []
        for i in range(len(words)):
            char_feats.append(torch.cat((char_hids[ix_seq[i]],char_hids[ix_seq[i+1]-1]),dim=1).view((1,1,-1)))
        char_feats = torch.cat(char_feats,dim=1).view((len(char_feats),1,-1))
        embeds = torch.cat((char_feats,word_embeds),dim=2)
        hiddens = self.lstm(embeds,self._init_hidden())[0]
        outputs = self.hid2tag(hiddens)
        return outputs

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
        
        feats = self._get_lstm_features(sent).view((len(sent[0]),-1))
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

WORD_EMBED_DIM = 90
CHAR_EMBED_DIM = 100
CHAR_HID_DIM = 160
HID_DIM = 120

model = ConcatCharLSTM_LSTM_CRF(VOCAB_SZ,WORD_EMBED_DIM,CHARSET_SZ,CHAR_EMBED_DIM,CHAR_HID_DIM,HID_DIM,tag2ix)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

with torch.no_grad():
    print(train_data[0][1])
    print(model(train_data[0][0]))
    print(model.loss(train_data[0][0],train_data[0][1]))

print("Test Successfully.")

file = open("concat_charlstm_lstm_crf.txt",mode="a+")
file.write("Char Embed Dim: %d  Char Hidden Dim: %d  Word Embed Dim: %d  Hidden Dim: %d  \n"%(CHAR_EMBED_DIM,CHAR_HID_DIM,WORD_EMBED_DIM,HID_DIM))
file.close()



# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(5): # again, normally you would NOT do 300 epochs, it is toy data
    print(epoch)
    import time
    t1 = time.time()
    loss_t = 0
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        # Step 3. Run our forward pass.
        loss = model.loss(sentence, tags)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        loss_t += loss.item()
        optimizer.step()
    t2 = time.time()
    file = open("concat_charlstm_lstm_crf.txt", mode="a+")
    file.write("epoch: %d  "%(epoch))
    file.write("train time: %f s  "%((t2-t1)/60))
    file.write("total loss: %f\n"%(loss_t))
    if epoch%1 == 0:
        with torch.no_grad():
            true, pred = [], []
            for sentence, tags in test_data:
                true.append([ix2tag[tag.item()] for tag in tags])
                pred.append([ix2tag[ix] for ix in model(sentence)])
        file.write("Accuracy: %f  Precision: %f  Recall Rate: %f  F_measure: %f\n"%evaluate(pred,true))
    file.close()
    #torch.save(model,'.\\CONCAT_CHARLSTM_BiLSTM_CRF_' + str(WORD_EMBED_DIM) + '_' + str(CHAR_EMBED_DIM) + '_' + str(CHAR_HID_DIM) + '_' + str(HID_DIM))