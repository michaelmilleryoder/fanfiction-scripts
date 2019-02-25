import os
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import shutil
from collections import Counter
import argparse
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from urllib.error import HTTPError
import re

""" Filter fics, make a train/dev/test split 
    Save out train/dev/test fic_ids, metadata and text
"""

def build_normalization_dict(metadata, input_tag_colname, out_dirpath):

    # Get all tags
    tags = sorted(set([t for l in metadata[input_tag_colname].values.tolist() for t in l]))

    # ## Build normalization dict

    metatags = {} # tag: [metatag/canonical tag] for tag normalization

    url_base = 'https://archiveofourown.org/tags/{}'
    for tag in tqdm(tags):

        orig_tag = tag
        
        #tag = re.sub(r'[/\.]', '*d*', tag)
        tag = tag.replace('/', '*s*').replace('.', '*d*')
        url = url_base.format(urllib.parse.quote(tag, safe=''))
        try:
            page = str(urllib.request.urlopen(url).read().decode('utf-8'))
        except HTTPError as e:
            continue
        
        if '<h3 class="heading">Mergers</h3>' in page: # has been merged
            soup = BeautifulSoup(page, 'html.parser') 
            canonical_tag = soup.find('div', {'class': 'merger module'}).p.a.text
            
            tag = canonical_tag # check if canonical tag has meta tag
            tag = tag.replace('/', '*s*').replace('.', '*d*')
            url = url_base.format(urllib.parse.quote(tag, safe=''))
            try:
                page = str(urllib.request.urlopen(url).read().decode('utf-8'))
            except HTTPError as e:
                print(tag)
                print(url)
                pdb.set_trace()
            
        if 'Meta tags:' in page:
            soup = BeautifulSoup(page, 'html.parser') 
            
            all_metatags = [el.text for el in soup.find('h3', string='Meta tags:').next_sibling.next_sibling.find_all('a', {'class': 'tag'})]
            toplevel_metatags = set([el.a.text for el in soup.find('h3', string='Meta tags:').next_sibling.next_sibling.find_all('ul')])
            rm_indices = [i-1 for i, tag in enumerate(all_metatags) if tag in toplevel_metatags]

            # # Remove bottom-level metatags if have a corresponding toplevel
            metatags[orig_tag] = set([tag for i, tag in enumerate(all_metatags) if not i in rm_indices])

        else: # already top-level tag
            metatags[orig_tag] = set([tag])
        
    normalized_tags = set([x for l in metatags.values() for x in l])
    print(f"\tNumber of normalized tags: {len(normalized_tags)}")

    # Save normalization dict
    with open(normalization_dict_fpath, 'wb') as f:
        pickle.dump(metatags, f)

    return metatags


def normalize_tags(metadata, metatags, input_tag_colname, normalized_tag_colname, tag_threshold):
    metadata[normalized_tag_colname] = metadata[input_tag_colname].map(lambda x: [t for l in [metatags[tag] for tag in eval(x) if isinstance(x, str) and tag in metatags] for t in l])

    return metadata


def initial_filter(metadata, lower_word_limit, upper_word_limit):
    filtered_metadata = metadata[(metadata['words'] >= lower_word_limit) & \
                (metadata['words'] <= upper_word_limit) & \
                (metadata['language'] == 'English') & \
                (metadata['additional tags'].map(lambda x: len(x) > 2))]

    return filtered_metadata


def get_top_tags(metadata, input_tag_colname, tag_threshold, out_dirpath):

    """ Assume input_tag_colname has lists """

#    tags = metadata.set_index('fic_id').to_dict()['additional tags']
#    tags = {key: [tag for tag in eval(val)] for key,val in tags.items()}

    # Get top n tags
    tag_ctr = Counter([tag for l in metadata[input_tag_colname].values.tolist() for tag in l])
    tag_vocab = [a for a,b in tag_ctr.most_common(tag_threshold)]

    # Save out top n tags
    tag_outpath = os.path.join(os.path.join(out_dirpath, f'top{tag_threshold}_tags.txt'))
    with open(tag_outpath, 'w') as f:
        for tag, count in tag_ctr.most_common(tag_threshold):
            f.write(f'{tag}\t{count}\n')
    
    return tag_vocab, metadata


def tag_filter(metadata, input_tag_colname, filtered_tag_colname, tag_vocab):
    """ Assumes input tag colname contains lists """

    # New column in metadata
    metadata[filtered_tag_colname] = metadata[input_tag_colname].map(lambda x: [tag for tag in x if tag in tag_vocab])

    # Filter
    metadata = metadata[metadata[filtered_tag_colname].map(lambda x: len(x) > 0)]

    return metadata


def split(metadata, fic_ids, tag_colname, tag_vocab, out_dirpath):

    np.random.seed(9)
    np.random.shuffle(fic_ids)

    # Split
    fic_ids['train'], fic_ids['dev'], fic_ids['test'] = np.split(fic_ids['all'], [int(.8*len(fic_ids['all'])), int(.9*len(fic_ids['all']))])

    # Save out fic ids
    for fold in ['train', 'dev', 'test']:
        with open(os.path.join(out_dirpath, f'fic_ids_{fold}.txt'), 'w') as f:
            for fic_id in sorted(fic_ids[fold]):
                f.write(str(fic_id)+'\n')

    # Split metadata
    metadata_split = {}
    for fold in ['train', 'dev', 'test']:
        metadata_split[fold] = metadata[metadata['fic_id'].isin(fic_ids[fold])].copy()
    print(f'Number of words in training set: {metadata_split["train"]["words"].sum()}')

    # Ensure all tags are present in all folds at least once
    for fold in ['train', 'dev', 'test']:
        fold_tags = set([t for l in metadata_split[fold][tag_colname] for t in l])

        for tag in tag_vocab:
            assert tag in fold_tags

    # Save metadata CSV
    for fold in ['train', 'dev', 'test']:
        fpath = os.path.join(out_dirpath, f'metadata_{fold}.csv')
        metadata_split[fold].to_csv(fpath, index=False)

    return metadata_split


def copy_fics(fic_ids, fandom_dirpath, out_dirpath):
    for fold in ['train', 'dev', 'test']:
        for tokenization in ['sent', 'para']:
            fold_out_dirpath = os.path.join(out_dirpath, f'{fold}_{tokenization}s')
            if not os.path.exists(fold_out_dirpath):
                os.mkdir(fold_out_dirpath)

            for fic_id in fic_ids[fold]:
                if tokenization == 'para':
                    fname = f"{fic_id}_tokenized_paras.txt"
                else:
                    fname = f"{fic_id}.txt"

                shutil.copy(os.path.join(fandom_dirpath, f'fics_{tokenization}s', fname), os.path.join(fold_out_dirpath, f"{fic_id}.txt"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('fandom', nargs='?', help='Name of fandom')
    parser.add_argument('tag_threshold', nargs='?', help='Number of top tags to keep')
    parser.add_argument('dataset_name', nargs='?', help='Name of dataset')
    args = parser.parse_args()

    # Settings
    fandom = args.fandom #'song_ice_fire',
    dataset_name = args.dataset_name
    tag_threshold = int(args.tag_threshold)
    #tag_search = [r'AU', r'(A|a)lternate (U|u)niverse', r'(C|c)anon divergen']
    #tag_replacement = 'AU'
    upper_word_limit = 5000
    lower_word_limit = 1000
    input_tag_colname = 'additional tags'
    normalized_tag_colname = 'normalized_tags'
    overwrite_tag_normalization_dict = True
    normalize = True
    if normalize:
        top_tag_colname = f'top{tag_threshold}_normalized_tags'
    else:
        top_tag_colname = f'top{tag_threshold}_tags'

    # I/O
    metadata_fpath = f'/usr2/scratch/fanfic/ao3_{fandom}_text/stories.csv'
    fandom_dirpath = f'/usr2/mamille2/fanfiction-project/data/ao3/{fandom}/'
    out_dirpath = os.path.join(fandom_dirpath, dataset_name)
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)
    normalization_dict_fpath = os.path.join(out_dirpath, 'tag_normalization.pkl')

    # Load metadata
    metadata = pd.read_csv(metadata_fpath)

    print(fandom)
    print(f'Found {len(metadata)} fics')

    # Make sure tag colname has lists, not strings
    if isinstance(metadata.iloc[0][input_tag_colname], str):
        metadata[input_tag_colname] = metadata[input_tag_colname].map(lambda x: eval(x))

    # Filter
    metadata = initial_filter(metadata, lower_word_limit, upper_word_limit)

    # Normalize tags
    if normalize: 
        if not os.path.exists(normalization_dict_fpath) or overwrite_tag_normalization_dict:
            print("Building normalization dictionary...")
            metatags = build_normalization_dict(metadata, input_tag_colname, out_dirpath)

        else:
            print("Loading normalization dictionary...")
            with open(normalization_dict_fpath, 'rb') as f:
                metatags = pickle.load(f)

        print("Normalizing tags...")
        metadata = normalize_tags(metadata, input_tag_colname, normalized_tag_colname, metatags, tag_threshold)

        input_tag_colname = normalized_tag_colname

    # Calculate top tags
    tag_vocab, metadata = get_top_tags(metadata, input_tag_colname, tag_threshold, out_dirpath)

    # Restrict dataset to having at least one tag in threshold
    metadata = tag_filter(metadata, input_tag_colname, top_tag_colname, tag_vocab)

    fic_ids = {}
    fic_ids['all'] = metadata['fic_id'].values.tolist()
    print(f'Filtered to {len(fic_ids["all"])} fics')

    # Shuffle and split
    metadata_split = split(metadata, fic_ids, top_tag_colname, tag_vocab, out_dirpath)
    for fold in ['train', 'dev', 'test']:
        print(f'{fold} set: {len(metadata_split[fold])} fics')


    # Copy fics
    copy_fics(fic_ids, fandom_dirpath, out_dirpath)

    # Save dataset parameters, info
    with open(os.path.join(out_dirpath, 'info.txt'), 'w') as f:
        f.write(f'Lower word limit: {lower_word_limit}\n')
        f.write(f'Upper word limit: {upper_word_limit}\n')
        f.write(f'Tag threshold: {tag_threshold}\n')
        f.write(f'Total fics: {len(fic_ids["all"])}\n')
        for fold in ['train', 'dev', 'test']:
            f.write(f'{fold} set: {len(metadata_split[fold])} fics\n')


if __name__ == '__main__':
    main()
