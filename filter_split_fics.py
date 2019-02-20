import os
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm_notebook as tqdm
import shutil
from collections import Counter

""" Filter fics, make a train/dev/test split 
    Save out train/dev/test fic_ids, metadata and text
"""

fandom = 'harrypotter'
metadata_fpath = f'/usr2/scratch/fanfic/ao3_{fandom}_text/stories.csv'
para_dirpath = f'/usr2/mamille2/fanfiction-project/data/ao3/{fandom}/'

# Load metadata
metadata = pd.read_csv(metadata_fpath)

print(fandom)
print(f'Found {len(metadata)} fics')

# Filter
upper_word_limit = 5000
lower_word_limit = 1000

metadata = metadata[(metadata['words'] >= lower_word_limit) & \
            (metadata['words'] <= upper_word_limit) & \
            (metadata['additional tags'].map(lambda x: len(x) > 2))]

fic_ids = {}
fic_ids['all'] = metadata['fic_id'].values.tolist()
print(f'Filtered to {len(fic_ids["all"])} fics')
print(f'Number of words: {metadata["words"].sum()}')

# Shuffle
np.random.shuffle(fic_ids)

# Split
fic_ids['train'], fic_ids['dev'], fic_ids['test'] = np.split(fic_ids['all'], [int(.8*len(fic_ids['all'])), int(.9*len(fic_ids['all']))])
for fold in ['train', 'dev', 'test']:
    print(f'{fold} set: {len(fic_ids[fold])} fics')

# Save out fic ids
for fold in ['train', 'dev', 'test']:
    with open(os.path.join(para_dirpath, f'fic_ids_{fold}.txt'), 'w') as f:
        for fic_id in sorted(fic_ids[fold]):
            f.write(str(fic_id)+'\n')

# Split metadata
metadata_split = {}
for fold in ['train', 'dev', 'test']:
    metadata_split[fold] = metadata[metadata['fic_id'].isin(fic_ids[fold])].copy()

# Calculate top tags
tags = {}

for fold in ['train', 'dev']:
    tags[fold] = metadata_split[fold].set_index('fic_id').to_dict()['additional tags']
    tags[fold] = {key: [tag for tag in eval(val)] for key,val in tags[fold].items()}

# Get top 100 tags
tag_ctr = Counter([tag for l in tags['train'].values() for tag in l])
tag_vocab = {}
tag_vocab[100] = [a for a,b in tag_ctr.most_common(100)]
tag_vocab[100]

# Add in column, save metadata CSV
for fold in ['train', 'dev', 'test']:
    metadata_split[fold]['top100_tags'] = metadata['additional tags'].map(lambda x: [tag for tag in eval(x) if tag in tag_vocab[100]])
    fpath = os.path.join(para_dirpath, f'metadata_{fold}.csv')
    metadata_split[fold].to_csv(fpath, index=False)

# Copy fics
for fold in ['train', 'dev', 'test']:

    out_dirpath = os.path.join(para_dirpath, fold)
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    for fic_id in fic_ids[fold]:
        fname = f"{fic_id}_tokenized_paras.txt"

        shutil.copy(os.path.join(para_dirpath, 'fics', fname), os.path.join(para_dirpath, fold, f"{fic_id}.txt"))
