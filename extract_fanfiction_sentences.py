import pandas as pd
from tqdm import tqdm
import os
from nltk.tokenize import sent_tokenize
import pdb

# Load fanfiction data
fandoms = [
    #'academia',
#    'allmarvel',
#    'detroit',
    #'friends',
    'supernatural',
]

data_dirpath_template = '/usr2/scratch/fanfic/ao3_{}_text/stories'
output_dirpath = '/usr2/mamille2/fanfiction-project/data/ao3/{0}'
output_fpath = os.path.join(output_dirpath, 'ao3_{0}_sentences.txt')

for f in fandoms:
    print(f)
    fandom_dirpath = data_dirpath_template.format(f)
    
    if not os.path.exists(output_dirpath.format(f)):
        os.makedirs(output_dirpath.format(f))

    with open(output_fpath.format(f), 'w') as fil:
        for fname in tqdm(os.listdir(fandom_dirpath)):
            paras = pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist()
            for para in paras:
                if not isinstance(para, str):
                    continue
                sents = sent_tokenize(para)
                for sent in sents:
                    fil.write(f"{sent}\n")
