import pandas as pd
from tqdm import tqdm
import os
from nltk.tokenize import sent_tokenize

# Load fanfiction data
fandoms = [
#    'academia',
#    'allmarvel',
#    'detroit',
    'friends',
]

data_dirpath_template = '/usr2/scratch/fanfic/ao3_{}_text/stories'
output_fpath = '/usr2/mamille2/fanfiction-project/data/ao3_{}_sentences.txt'

for f in fandoms:
    print(f)
    fandom_dirpath = data_dirpath_template.format(f)
    
    with open(output_fpath, 'w') as f:
        for fname in tqdm(os.listdir(fandom_dirpath)):
            paras = pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist()
            for para in paras:
                if not isinstance(para, str):
                    continue
                sents = sent_tokenize(para)
                for sent in sents:
                    f.write(sent)
