"""
This file, when executed, should train all needed model and save them locally. 
Assume all packages are available to use
"""
import torch
import tensorflow
import os, sys

print("STARTED FINISHED INSTALLING ALLEN")

# credit to https://www.kaggle.com/bkkaggle/allennlp-packages
os.system("cp -r ../input/allennlp-packages/packages/packages ./")
os.system("pip install -r packages/requirements.txt --no-index --find-links packages")
os.system("pip install ../input/sacremoses-tokenizer/sacremoses-master/sacremoses-master")
print("FINISHED INSTALLING ALLEN")
print("finished install moses")
from keras.preprocessing import text, sequence
import pandas as pd
import numpy as np
import torch.utils.data
import torch.nn as nn
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from sacremoses import MosesTokenizer
from sklearn.metrics import roc_auc_score
MAX_LEN = 50
SHUFFLE = True
EPOCH = 4
BATCH_SIZE = 16
ELMO_WEIGHT = "../input/pretrained-weights/weights/weights/elmo_small.hdf5"
ELMO_CONF = "../input/pretrained-weights/weights/weights/elmo_small.json"
WEIGHT_FILE = "../input/pretrained-weights/weights/weights/linear223000.th"
print("finished init imports")
"""
    Part 1: Data processing
"""
def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data
    def __getitem__(self, index):
        ret = []
        for i in self.data:
            ret.append(i[index])
        return ret
    def __len__(self):
        return len(self.data[0])
def data_loader_eval():
    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
    x_test = preprocess(test['comment_text'])
    datset = SimpleDataset(x_test, test["id"])
    return torch.utils.data.DataLoader(datset, batch_size=BATCH_SIZE*4)

"""
    Part 2: Model def and training
"""
def load_elmo():
    return ElmoEmbedder(options_file = ELMO_CONF, weight_file = ELMO_WEIGHT, cuda_device = 0)
class ElmoClassifier(nn.Module):
    def __init__(self):
        super(ElmoClassifier, self).__init__()
        self.elmo = load_elmo()
        self.tokenizer = MosesTokenizer(lang='en')
        self.fc_1 = nn.Linear(512*2, 1)
        #self.avg_pool = nn.AdaptiveAvgPool2d((200, 1024))
        #self.max_pool = nn.AdaptiveMaxPool2d((200, 1024))
        self.lstm_1 = nn.LSTM(768, 512, bidirectional=True, num_layers = 2, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        with torch.no_grad():
            x = [self.tokenizer.tokenize(sentence, escape=False) for sentence in x]
            x = self.elmo.batch_to_embeddings(x)[0]
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm_1(x)
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_1(x)
        return(x)

def eval(**kwargs):
    model = ElmoClassifier()
    model.load_state_dict(torch.load(kwargs["elmo"])["model_state_dict"])
    print("LOADED EVAL")
    model = model.cuda()
    model = model.eval()
    data_iter = data_loader_eval()
    sigmoid = nn.Sigmoid()
    id_total = []
    pred_total = []
    i =0
    for x, id_num in data_iter:
        with torch.no_grad():
            output = model.forward(x)
            output = sigmoid(output)
            pred_total += [i.item() for i in output.view(-1).cpu()] 
            id_total += [i.item() for i in id_num]
        if i %100== 0:
            print(i)
        i+=1
    submission = pd.DataFrame(data = {"id": id_total, "prediction": pred_total})
    submission.to_csv('submission.csv', index=False)

        
if __name__ == "__main__":
    sentences = [['First', 'sentence', '.'], ['Another', '.', "hello", "man"]]
    #train()
    print("STARTED EVAL")
    eval(elmo = WEIGHT_FILE)