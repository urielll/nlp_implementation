import os, sys
import pandas as pd
import gluonnlp as nlp

data = pd.read_table('./data/ratings_train.txt')
data['document'] = data['document'].fillna('')
