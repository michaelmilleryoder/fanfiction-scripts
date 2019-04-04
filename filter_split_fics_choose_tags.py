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
import re
from sklearn.model_selection import train_test_split
import pdb

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
        
        url = url_base.format(urllib.parse.quote(tag, safe=''))
        page = str(urllib.request.urlopen(url).read())
        
        if '<h3 class="heading">Mergers</h3>' in page: # has been merged
            soup = BeautifulSoup(page, 'html.parser') 
            canonical_tag = soup.find('div', {'class': 'merger module'}).p.a.text
            
            tag = canonical_tag # check if canonical tag has meta tag
            url = url_base.format(urllib.parse.quote(tag, safe=''))
            page = str(urllib.request.urlopen(url).read())
            
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
    metadata[normalized_tag_colname] = metadata[input_tag_colname].map(lambda x: [t for l in [metatags[tag] for tag in eval(x) if isinstance(x, str)] for t in l])

    return metadata


def initial_filter(metadata, lower_word_limit, upper_word_limit, fandom_dirpath, save=False):
    filtered_metadata = metadata[(metadata['words'] >= lower_word_limit) & \
                (metadata['words'] <= upper_word_limit) & \
                (metadata['language'] == 'English') & \
                (metadata['additional tags'].map(lambda x: len(x) > 2))]

    # Check for any empty fics
    for fic_id in filtered_metadata['fic_id']:
        fname = f"{fic_id}.txt"

        text_fpath = os.path.join(fandom_dirpath, f'fics_sents', fname)
        if os.stat(text_fpath).st_size == 0:
            filtered_metadata = filtered_metadata[filtered_metadata['fic_id'] != fic_id]

    if save:
        print("Saving initial filter fics...")
        for tokenization in ['sent', 'para']:
            fold_out_dirpath = os.path.join(fandom_dirpath, f'filtered_{tokenization}s')
            if not os.path.exists(fold_out_dirpath):
                os.mkdir(fold_out_dirpath)

            for fic_id in filtered_metadata['fic_id']:
                if tokenization == 'para':
                    fname = f"{fic_id}_tokenized_paras.txt"
                else:
                    fname = f"{fic_id}.txt"

                shutil.copy(os.path.join(fandom_dirpath, f'fics_{tokenization}s', fname), os.path.join(fold_out_dirpath, f"{fic_id}.txt"))

    return filtered_metadata


def get_selected_tags(metadata, input_tag_colname, tag_search, out_dirpath):

    """ Assume input_tag_colname has lists """

    # Get matching tags
    tags = set([tag for l in metadata[input_tag_colname].values.tolist() for tag in l])

    selected_tags = set()
    for p in tag_search:
        matching_tags = set([tag for tag in tags if re.match(p, tag)])
        selected_tags |= matching_tags

    # Save out selected tags
    tag_outpath = os.path.join(os.path.join(out_dirpath, f'selected_tags.txt'))
    with open(tag_outpath, 'w') as f:
        for tag in sorted(list(selected_tags)):
            f.write(f'{tag}\n')
    
    return selected_tags, metadata


def replace_tags(tags, tag_replacements):

    replacements = set()

    for tag in tags:
        for p in tag_replacements:
            if re.search(p, tag):
                replacements.add(tag_replacements[p])

    return list(replacements)


def select_tags(metadata, input_tag_colname, selected_tag_colname, tag_replacements):
    """ Assumes input tag colname contains lists """

    # New column in metadata
    metadata[selected_tag_colname] = [replace_tags(tags, tag_replacements) for tags in metadata[input_tag_colname].values.tolist()]

    return metadata


def split(metadata, tag_colname, tag_vocab, out_dirpath):

    fic_ids = {}
    fic_ids['all'] = metadata['fic_id'].values.tolist()
    print(f'Filtered to {len(fic_ids["all"])} fics')

    #np.random.seed(9)
    #np.random.shuffle(fic_ids)

    ## Split
    #fic_ids['train'], fic_ids['dev'], fic_ids['test'] = np.split(fic_ids['all'], [int(.8*len(fic_ids['all'])), int(.9*len(fic_ids['all']))])

    # Split metadata
    metadata_split = {}

    #metadata_split['train'], metadata_split['test'] = train_test_split(metadata, test_size=0.1, stratify=metadata[tag_colname])
    #metadata_split['train'], metadata_split['dev'] = train_test_split(metadata_split['train'], test_size=1/9, stratify=metadata_split['train'][tag_colname])

    metadata_split['train'], metadata_split['test'] = train_test_split(metadata, test_size=0.1, random_state=9)
    metadata_split['train'], metadata_split['dev'] = train_test_split(metadata_split['train'], test_size=1/9, random_state=9)

    #for fold in ['train', 'dev', 'test']:
        #metadata_split[fold] = metadata[metadata['fic_id'].isin(fic_ids[fold])].copy()
    #print(f'Number of words in training set: {metadata_split["train"]["words"].sum()}')

    # Ensure all tags are present in all folds at least once
    for fold in ['train', 'dev', 'test']:
        fold_tags = set([t for l in metadata_split[fold][tag_colname] for t in l])

        for tag in tag_vocab:
            assert tag in fold_tags
            tag_proportion = sum(metadata_split[fold][tag_colname].map(lambda x: tag in x))/len(metadata_split[fold])
            #print(f"{fold} pos examples of {tag}: {tag_proportion}")

    # Save out fic ids
    fic_ids = {}
    for fold in ['train', 'dev', 'test']:
        fic_ids[fold] = metadata_split[fold]['fic_id'].values.tolist()
        with open(os.path.join(out_dirpath, f'fic_ids_{fold}.txt'), 'w') as f:
            for fic_id in sorted(fic_ids[fold]):
                f.write(str(fic_id)+'\n')

    # Save metadata CSV
    for fold in ['train', 'dev', 'test']:
        fpath = os.path.join(out_dirpath, f'metadata_{fold}.csv')
        metadata_split[fold].to_csv(fpath, index=False)

    return metadata_split, fic_ids


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

def sample_negatives(metadata, selected_tag_colname, sampling_strategy):

    positives = metadata[metadata[selected_tag_colname].map(lambda x: len(x) > 0)]

    negatives = metadata[metadata[selected_tag_colname].map(lambda x: len(x) == 0)]


    if sampling_strategy == 'average':
        n_possible_values = len(set([t for l in metadata[selected_tag_colname] for t in l]))        

        n_negative_samples = int(len(positives)/n_possible_values)

    else: # equal strategy
        n_negative_samples = len(positives)

    sampled_negatives = negatives.sample(n=n_negative_samples, random_state=9)

    sampled = metadata.loc[positives.index.append(sampled_negatives.index)].copy()

    return sampled


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('fandom', nargs='?', help='Name of fandom')
    parser.add_argument('dataset_name', nargs='?', help='Name of dataset')
    args = parser.parse_args()

    # Settings
    fandom = args.fandom
    dataset_name = args.dataset_name
    #tag_search = [r'AU', r'(A|a)lternate (U|u)niverse', r'(C|c)anon divergen']
    #tag_search = [r'Fluff', r'Angst', r'Romance',
    #                r'Humor', r'Hurt/Comfort']
    tag_search = [r'(D|d)raco']
    tag_replacements = {re.compile(t, re.IGNORECASE): t for t in tag_search}
    negative_sampling_strategy = 'average'
    upper_word_limit = 5000
    lower_word_limit = 1000
    input_tag_colname = 'additional tags'
    save_initial_filter = False
    normalized_tag_colname = 'normalized_tags'
    overwrite_tag_normalization_dict = True
    normalize = False
    if normalize:
        selected_tag_colname = f'top{tag_threshold}_normalized_tags'
    else:
        #selected_tag_colname = f'top{tag_threshold}_tags'
        selected_tag_colname = 'selected_tags'

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
    metadata = initial_filter(metadata, lower_word_limit, upper_word_limit, fandom_dirpath, save_initial_filter)

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

    # add selected tags column
    metadata = select_tags(metadata, input_tag_colname, selected_tag_colname, tag_replacements)

    # Enforce positive class proportion
    metadata = sample_negatives(metadata, selected_tag_colname, negative_sampling_strategy)

    # Shuffle and split
    metadata_split, fic_ids = split(metadata, selected_tag_colname, tag_replacements.values(), out_dirpath)
    for fold in ['train', 'dev', 'test']:
        print(f'{fold} set: {len(metadata_split[fold])} fics')

    # Copy fics
    print("Copying fics...")
    copy_fics(fic_ids, fandom_dirpath, out_dirpath)

    # Save dataset parameters, info
    with open(os.path.join(out_dirpath, 'info.txt'), 'w') as f:
        f.write(f'Lower word limit: {lower_word_limit}\n')
        f.write(f'Upper word limit: {upper_word_limit}\n')
        f.write(f'Selected tags: {tag_search}\n')
        f.write(f'Negative sampling strategy: {negative_sampling_strategy}\n')
        f.write(f'Total fics: {len(metadata)}\n')
        for fold in ['train', 'dev', 'test']:
            f.write(f'{fold} set: {len(metadata_split[fold])} fics\n')
        f.write(f'Number of words in training set: {metadata_split["train"]["words"].sum()}')


if __name__ == '__main__':
    main()
