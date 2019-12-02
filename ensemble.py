# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# os.system("cp ../input/local-result/submission.csv submission.csv")
# Any results you write to the current directory are saved as output.

pp = pd.read_csv('../input/local-result/submission-preprocess.csv')
attn = pd.read_csv('../input/local-result/submission-attn.csv')
pools = pd.read_csv('../input/local-result/submission-pools.csv')
lstmcnn = pd.read_csv('../input/local-result/submission-lstm-cnn.csv')
pp_atten_pool = pd.read_csv("../input/local-result/submission-lstm-cnn-attention-pool.csv")
evol = pd.read_csv("../input/local-result/submission-cnn-lstm-gru-evolved.csv")
full_bert = pd.read_csv("../input/local-result/submission-full-cnn-lstm.csv")

pp["prediction"] = 1/7*pp["prediction"] + 1/7*attn["prediction"] +0.5/7*pools["prediction"] + 0.5/7*lstmcnn["prediction"] +3/7*pp_atten_pool["prediction"] + 0.5/7*evol["prediction"] + 0.5/7*full_bert["prediction"]
pp.set_index('id', inplace=True)
pp.to_csv('./submission.csv')
