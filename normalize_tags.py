import os
import pandas as pd
from collections import Counter
import urllib.request
import urllib.parse
import html
from bs4 import BeautifulSoup
import pickle
from tqdm import tqdm


def load_metadata(fandom_dirpath):
    # Load metadata split fic IDs
    # fic_ids = {}
    metadata = {}
    tags = {}

    # Get tags for folds
    for fold in ['train', 'dev', 'test']:
        
        metadata[fold] = pd.read_csv(os.path.join(fandom_dirpath, f'metadata_{fold}.csv'))
    

        tags[fold] = metadata[fold].set_index('fic_id').to_dict()['additional tags']
        tags[fold] = {key: [tag for tag in eval(val)] for key,val in tags[fold].items()}


    # Get top n tags (could also just load from saved list)
    tag_ctr = Counter([tag for l in tags['train'].values() for tag in l])
    tag_vocab = {}
    tag_vocab[100] = [a for a,b in tag_ctr.most_common(100)]

    return metadata, tag_vocab[100]


def build_normalization_dict(metadata, tag_vocab, fandom_dirpath):
# ## Build normalization dict

    metatags = {} # tag: [metatag/canonical tag] for tag normalization

    url_base = 'https://archiveofourown.org/tags/{}'
    for tag in tqdm(tag_vocab):

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
    with open(os.path.join(fandom_dirpath, 'tag_normalization.pkl'), 'wb') as f:
        pickle.dump(metatags, f)

    return metatags


def normalize_tags(metadata, metatags, fandom_dirpath):
    for fold in ['train', 'dev', 'test']:
        metadata[fold]['top100_tags_normalized'] = metadata[fold]['top100_tags'].map(lambda x: [t for l in [metatags[tag] for tag in eval(x) if isinstance(x, str)] for t in l])

        metadata[fold].to_csv(os.path.join(fandom_dirpath, f'metadata_{fold}.csv'), index=False)


def main():

    #fandom = 'song_ice_fire'
    fandom = 'harrypotter'
    fandom_dirpath = f'/usr2/mamille2/fanfiction-project/data/ao3/{fandom}/'

    # See if already normalized
    print("Loading metadata...")
    metadata, tag_vocab = load_metadata(fandom_dirpath)
    if not 'top100_tags_normalized' in metadata['train'].columns:
        print("Building normalization dictionary...")
        metatags = build_normalization_dict(metadata, tag_vocab, fandom_dirpath)
        print("Normalizing tags...")
        normalize_tags(metadata, metatags, fandom_dirpath)


if __name__ == '__main__':
    main()
