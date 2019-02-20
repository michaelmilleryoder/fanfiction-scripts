import os
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm_notebook as tqdm
import shutil
from collections import Counter
import argparse

""" Filter fics, make a train/dev/test split 
    Save out train/dev/test fic_ids, metadata and text
"""

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('fandom', nargs='?', help='Name of fandom')
parser.add_argument('tokenization', nargs='?', help='Sentence or paragraph written per line')
args = parser.parse_args()

fandom = args.fandom #'song_ice_fire',
tokenization = args.tokenization # 'sent' or 'para'
tag_threshold = 100

metadata_fpath = f'/usr2/scratch/fanfic/ao3_{fandom}_text/stories.csv'
text_dirpath = f'/usr2/mamille2/fanfiction-project/data/ao3/{fandom}/'


def main():

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

    # Shuffle
    np.random.seed(9)
    np.random.shuffle(fic_ids)

    # Split
    fic_ids['train'], fic_ids['dev'], fic_ids['test'] = np.split(fic_ids['all'], [int(.8*len(fic_ids['all'])), int(.9*len(fic_ids['all']))])
    for fold in ['train', 'dev', 'test']:
        print(f'{fold} set: {len(fic_ids[fold])} fics')

    # Save out fic ids
    for fold in ['train', 'dev', 'test']:
        with open(os.path.join(text_dirpath, f'fic_ids_{fold}.txt'), 'w') as f:
            for fic_id in sorted(fic_ids[fold]):
                f.write(str(fic_id)+'\n')

    # Split metadata
    metadata_split = {}
    for fold in ['train', 'dev', 'test']:
        metadata_split[fold] = metadata[metadata['fic_id'].isin(fic_ids[fold])].copy()
    print(f'Number of words in training set: {metadata_split["train"]["words"].sum()}')

    # Calculate top tags
    tags = {}

    for fold in ['train', 'dev']:
        tags[fold] = metadata_split[fold].set_index('fic_id').to_dict()['additional tags']
        tags[fold] = {key: [tag for tag in eval(val)] for key,val in tags[fold].items()}

    # Get top n tags
    tag_ctr = Counter([tag for l in tags['train'].values() for tag in l])
    tag_vocab = {}
    tag_vocab[tag_threshold] = [a for a,b in tag_ctr.most_common(tag_threshold)]
    tag_vocab[tag_threshold]

    # Save out top n tags
    tag_outpath = os.path.join(os.path.join(text_dirpath, f'top_tags_{tag_threshold}.txt'))
    with open(tag_outpath, 'w') as f:
        for tag, count in tag_ctr.most_common(tag_threshold):
            f.write(f'{tag}\t{count}\n')

    # Add in column, save metadata CSV
    for fold in ['train', 'dev', 'test']:
        metadata_split[fold]['top100_tags'] = metadata['additional tags'].map(lambda x: [tag for tag in eval(x) if tag in tag_vocab[100]])
        fpath = os.path.join(text_dirpath, f'metadata_{fold}.csv')
        metadata_split[fold].to_csv(fpath, index=False)

    # Copy fics
    for fold in ['train', 'dev', 'test']:

        out_dirpath = os.path.join(text_dirpath, f'{fold}_{tokenization}s')
        if not os.path.exists(out_dirpath):
            os.mkdir(out_dirpath)

        for fic_id in fic_ids[fold]:
            if tokenization == 'para':
                fname = f"{fic_id}_tokenized_paras.txt"
            else:
                fname = f"{fic_id}.txt"

            shutil.copy(os.path.join(text_dirpath, f'fics_{tokenization}s', fname), os.path.join(out_dirpath, f"{fic_id}.txt"))

if __name__ == '__main__':
    main()
