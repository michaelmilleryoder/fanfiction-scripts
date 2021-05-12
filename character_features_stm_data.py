#!/usr/bin/env python
# coding: utf-8

""" Create character feature dataset for STM 
    Annotate fic relationship type: straight or queer, by inferring character gender. 
    Also matches characters in relationship list to pipeline characters
@author Michael Miller Yoder
@year 2021
"""
import re
import json
import os
import math
import pdb
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd


def remove_charnames(text, to_remove):
    """ Remove character names and stopwords from a string """
    toks = text.split()
    return ' '.join([t for t in toks if t.lower() not in to_remove])


def add_character_features_fandom(params):
    """ Add character features for a specific fandom """
    data, charfeats_dirpath, quote_dirpath = params
    
    # Build list of character names (to remove from assertion features)
    chars = {char for chars in data.character.map(to_list_column) for char in chars}
    char_parts = {char_part.lower() for char in chars for char_part in char.split() \
        if char_part[0].isupper()}
    stops = {'n’t', "n't", '’ve', "'ve", '’ll', "'ll", '’re', "'re", 'don’t', "don't"}
    to_remove = char_parts | stops
    
    # Assemble assertion character features
    header = ['fic_id', 'pipeline_character', 'assertion_features', 'quote_features']
    outlines = []
    for fname in os.listdir(charfeats_dirpath):
    #for fname in tqdm(os.listdir(charfeats_dirpath)):
        fic_id = int(fname.split('.')[0])
        with open(os.path.join(charfeats_dirpath, fname)) as f:
            char_features = json.load(f)
        with open(os.path.join(quote_dirpath, f'{fic_id}.quote.json')) as f:
            quote_features = json.load(f)
        for char in char_features:
            feats = ' '.join([w for w in char_features[char] if w.lower() \
                not in char_parts])
            quotes = ''
            if char in quote_features:
                quotes = ' '.join([remove_charnames(q['text'], to_remove) for q \
                    in quote_features[char]])
            if len(feats) < 1 and len(quotes) < 1: 
                continue
            outlines.append([fic_id, char, feats, quotes])
    char_feats = pd.DataFrame(outlines, columns=header).set_index('fic_id')

    # Explode to characters
    data = data.explode('coref_char_matches')
    data = pd.merge(data, char_feats, left_index=True, right_index=True) # just on fic_id

    # Filter to just those characters present in coref_char_matches
    cond = [char in matches for char, matches in zip(data.pipeline_character,
         data.coref_char_matches)]
    data = data[cond]

    # Combine character features for characters that match the same character in a relationship
    agg =  {'assertion_features': ' '.join,
            'quote_features': ' '.join}
    # cols to keep
    agg.update({col: 'first' for col in [
        'title', 'rating', 'category', 'relationship', 'kudos', 
        'pipeline_character', 'relationship_type', 'published',
        'fandom', 'dataset']})
    data = data.groupby([data.index, data['coref_char_matches'].astype(str)]).agg(
        agg)

    # Filter to fics with characters only in one relationship
    # Fics with 1 seem to be errors/rare cases where a character appears in coref 
    # (and thus matches the relationship) but doesn't have assertions
    # Fics with other odd numbers are where the same character is involved in 
    # multiple relationships (maybe put queer then? remove)
    sizes = data.groupby('fic_id').size()
    inds = sizes[sizes%2==0].index.tolist()
    data = data.iloc[data.index.get_level_values(
        'fic_id').isin(inds)]

    return data


def match_pipeline_chars_fandom(params):
    """ Match AO3 metadata characters to characters in pipeline output,
        for just one fandom.
        Also annotates relationship type
    """
    data, coref_dirpath = params
    data['coref_char_matches'] = [match_coref_chars(fic_id, rel, 
        coref_dirpath) for fic_id, rel in list(
        zip(data.index, data['relationships']))]

    # Filter out relationships with no matching characters for at least 1 char
    data = data[data['coref_char_matches'].map(
        lambda x: len(x[0])>=1 and len(x[1])>=1)].copy()

    # Annotate relationship type
    data['relationship_type'] = [annotate_relationship_gender(fic_id, rel, 
        coref_dirpath) for fic_id, rel in list(zip(
        data.index, data.coref_char_matches))]

    return data

def to_list_column(string):
    return string[2:-2].split('", "')


def annotate_relationship_gender(fic_id, rel, coref_dirpath):
    """ Annotate character pipeline names for genders, 
        Return straight if F/M, queer otherwise (including nonbinary) """
    # Load coref 
    coref_fpath = os.path.join(coref_dirpath, f'{fic_id}.json')
    with open(coref_fpath) as f:
        coref = json.load(f)
    char1, char2 = rel
    char1_mentions = [m for mentions in [cluster['mentions'] for cluster in coref['clusters'] if 'name' in cluster and cluster['name'] in char1] for m in mentions]
    char2_mentions = [m for mentions in [cluster['mentions'] for cluster in coref['clusters'] if 'name' in cluster and cluster['name'] in char2] for m in mentions]
    char1_mentions = [m['text'] if 'text' in m else m['phrase'] for m in char1_mentions]
    char2_mentions = [m['text'] if 'text' in m else m['phrase'] for m in char2_mentions]
    char1_gender = infer_gender(char1_mentions)
    char2_gender = infer_gender(char2_mentions)
    
    label = 'queer'
    if (char1_gender == 'F' and char2_gender == 'M') or (
        char2_gender == 'F' and char1_gender == 'M'):
        label = 'straight'
    return label


def infer_gender(mentions):
    """ Infer gender based on pronouns in character mentions """
    male_pronouns = ['he', 'him', 'his']
    female_pronouns = ['she', 'her', 'hers']
    nonbinary_pronouns = ['they', 'them', 'their', 'theirs']
    lowered_mentions = [m.lower() for m in mentions]
    male_count = sum(lowered_mentions.count(p) for p in male_pronouns)
    female_count = sum(lowered_mentions.count(p) for p in female_pronouns)
    nonbinary_count = sum(lowered_mentions.count(p) for p in nonbinary_pronouns)
    gender = 'UNK'
    if male_count > female_count and male_count > nonbinary_count:
        gender = 'M'
    elif female_count > male_count and female_count > nonbinary_count:
        gender = 'F'
    elif nonbinary_count > male_count and nonbinary_count > female_count:
        gender = 'NB'
    return gender


def characters_match(char1, char2):
    """ Returns True if 2 character names match closely enough.
        First splits character names into parts by underscores or spaces.
        Matches either if:
            * Any part matches and either name has only 1 part (Potter and Harry Potte    r, e.g.)
            * The number of part matches is higher than half of unique name parts betw    een the 2 characters
        From /projects/fanfiction-nlp-evaluation/annotated_span.py
    """
    honorifics = ['ms.', 'ms',
                    'mr.', 'mr',
                    'mrs.', 'mrs',
                    'uncle', 'aunt',
                    'dear', 'sir', "ma'am"
                ]
    char1_processed = re.sub(r'[\(\),]', '', char1)
    char1_parts = [part for part in re.split(r'[ _]', char1_processed.lower()) if not     part in honorifics]
    char2_processed = re.sub(r'[\(\),]', '', char2)
    char2_parts = [part for part in re.split(r'[ _]', char2_processed.lower()) if not     part in honorifics]

    # Count number of part matches
    n_parts_match = 0
    for part1 in char1_parts:
        for part2 in char2_parts:
            #if part1 == part2 and len(char1_parts)==1 or len(char2_parts)==1:
            if part1 == part2:
                n_parts_match += 1

    # Determine match
    not_surnames = ['male', 'female']
    if n_parts_match == 1 and (len(char1_parts) == 1 or len(char2_parts) == 1) and not     any([w in char1_parts for w in not_surnames]) and not any([w in char2_parts for w in     not_surnames]):
        match = True
    elif n_parts_match > len(set(char1_parts + char2_parts))/2:
        match = True
    else:
        match = False

    return match


def match_coref_chars(fic_id, rel, coref_dirpath):
    """ Try to find character matches from pipeline output.
        Returns a list of matching characters in the same order as listed in the relationship.
    """
    rel_chars = set(re.split(r'[\/]', rel))
    if len(rel_chars) != 2: # Only handle 2-partner relationships as of now
        return [[], []]
    
    coref_fpath = os.path.join(coref_dirpath, f'{fic_id}.json')
    if not os.path.exists(coref_fpath):
        return [[], []]
    with open(coref_fpath) as f:
        coref = json.load(f)
    coref_names = [cluster['name'] for cluster in coref['clusters'] if 'name' in cluster]
    
    rel_matches = list()
    for char in rel_chars:
        matches = list({coref_char for coref_char in coref_names if characters_match(char, coref_char)})
        if len(matches) >= 1:
            rel_matches.append(matches)
        else:
            rel_matches.append([])
    return rel_matches


class CharacterFeaturesDataset:
    """ Create and hold a dataset of character features, across fandoms """

    def __init__(self, fandoms, base_dirpath, dataset_name, output_dirpath):
        """ Args:
                fandoms: list of fandom names to include
                base_dirpath: <base_dirpath>/<fandom>/<dataset_name>/metadata.csv
                dataset_name: subdirectory within the <fandom> directory to find 
                    metadata.csv
                n_threads: number of threads to do processing
        """
        self.fandoms = fandoms
        self.base_dirpath = base_dirpath
        self.dataset_name = dataset_name
        self.coref_dirpaths = {fandom: os.path.join(base_dirpath, fandom, 
            self.dataset_name, 'output/char_coref') for fandom in self.fandoms}
        self.charfeats_dirpaths = {fandom: os.path.join(base_dirpath, fandom, 
            self.dataset_name, 'output/char_features') for fandom in self.fandoms}
        self.assertion_dirpaths = {fandom: os.path.join(base_dirpath, fandom, 
            self.dataset_name, 'output/assertion_extraction') for fandom in \
            self.fandoms}
        self.quote_dirpaths = {fandom: os.path.join(base_dirpath, fandom, 
            self.dataset_name, 'output/quote_attribution') for fandom in self.fandoms}
        self.check_character_features()
        self.outpath = os.path.join(output_dirpath, 
            f'charfeats_{len(self.fandoms)}fandoms.csv')
        self.data = None
        self.metadata_parts = {} # one for each fandom, for multiprocessing
        self.metadata = None # unified metadata parts

    def check_character_features(self):
        """ For all fandoms, check if character features have been extracted
            Save fandoms that have character features in self.fandoms
        """
        fandoms_with_feats = []
        for fandom in self.fandoms:
            if os.path.exists(self.charfeats_dirpaths[fandom]):
                fandoms_with_feats.append(fandom)
            elif os.path.exists(self.assertion_dirpaths[fandom]):
                print(f"{fandom} hasn't had assertion features extracted. "
                    "Run extract_character_features_from_assertions.py")
            else:
                print(f'No pipeline output found for {fandom}')
        self.fandoms = fandoms_with_feats

    def load_metadata(self):
        """ Load metadata for all fandoms and make every row a relationship in a fic.
            Save in self.metadata
        """
        dataset_fandoms = {
            'homestuck': 'hi_lgbtq',
            'startrek': 'hi_lgbtq',
            'dragonage': 'hi_lgbtq',
            'buffy': 'hi_lgbtq',
            'jojo': 'hi_lgbtq',
            'pokemon': 'hi_lgbtq',
            'danganronpa': 'hi_lgbtq',
            'glee': 'hi_lgbtq',
            'fire_emblem': 'hi_lgbtq',
            'hannibal': 'hi_lgbtq',
            'dcu': 'lo_lgbtq',
            'titan': 'lo_lgbtq',
            'harrypotter': 'lo_lgbtq',
            'percy_jackson': 'lo_lgbtq',
            'tolkien': 'lo_lgbtq',
            'naruto': 'lo_lgbtq',
            'teenwolf': 'lo_lgbtq',
            'song_ice_fire': 'lo_lgbtq',
            'shadowhunters': 'lo_lgbtq',
            'walking_dead': 'lo_lgbtq'
            }
        # Load fic metadata
        for fandom in self.fandoms:
            metadata_fpath = \
                f'/data/fanfiction_ao3/{fandom}/{self.dataset_name}/metadata.csv'
            metadata = pd.read_csv(metadata_fpath, index_col='fic_id', 
                parse_dates=['published'])
            range_beg = datetime(2010,1,1)
            range_end = datetime(2020,12,31)
            metadata = metadata[(metadata['published']>=range_beg) & (metadata[
                'published']<=range_end)]
            metadata.rename(columns={'fandom': 'fandoms'}, inplace=True)
            metadata['fandom'] = fandom
            metadata['dataset'] = dataset_fandoms[fandom]

            # Expand by relationship
            metadata['relationships'] = metadata['relationship'].map(to_list_column)
            rel_metadata = metadata.explode('relationships')

            # Filter to fics with romantic relationships
            rel_metadata = rel_metadata.dropna(subset=['relationships'])
            rel_metadata = rel_metadata[rel_metadata['relationships'].map(
                lambda x: '/' in x or '\\' in x)]
            self.metadata_parts[fandom] = rel_metadata
        #self.metadata = pd.concat(dfs)

    def match_pipeline_chars(self):
        """ Match AO3 metadata characters to characters from the pipeline 
            Output: self.data
        """
        params = [(self.metadata_parts[fandom], self.coref_dirpaths[fandom]) for \
            fandom in self.fandoms]
        with Pool(len(self.fandoms)) as p:
            parts = list(tqdm(p.imap(match_pipeline_chars_fandom, params), 
                ncols=70, total=len(self.fandoms)))

        # for debugging
        #params = [(self.metadata_parts[fandom], self.coref_dirpaths[fandom]) for \
        #    fandom in self.fandoms[:2]]
        #parts = list(map(match_pipeline_chars_fandom, tqdm(params, ncols=70, total=2)))
        self.metadata_parts = dict(zip(self.fandoms, parts))

    def add_character_features(self, assertions=True, quotes=False):
        """ Add in assertions and quotes for characters """
        params = [(self.metadata_parts[fandom], self.charfeats_dirpaths[fandom], 
                    self.quote_dirpaths[fandom]) for fandom in self.fandoms]

        with Pool(len(self.fandoms)) as p:
            parts = list(tqdm(p.imap(add_character_features_fandom, params), 
                ncols=70, total=len(self.fandoms)))

        # for debugging
        #parts = list(map(add_character_features_fandom, tqdm(params, ncols=70)))

        self.metadata_parts = dict(zip(self.fandoms, parts))

    def transform(self):
        """ Create dataset for STM """
        self.load_metadata()
        print('Matching to pipeline characters...')
        self.match_pipeline_chars()
        print('Adding in character features...')
        self.add_character_features()

    def save(self):
        """ Save out for STM """

        # Sample uniformly from each fandom
        min_len = math.floor(min(
            [len(data) for data in self.metadata_parts.values()])/100)*100
        print(f"Sampling {min_len} character rows from each fandom")

        # Combine fandom parts into output df
        self.data = pd.concat([data.sample(min_len) for data in \
            self.metadata_parts.values()])
        self.data.to_csv(self.outpath)
        print(f"Saved STM data with {len(self.data)} rows to " + self.outpath)


def main():
    """ Run the script """

    fandoms = [
        'homestuck',
        'startrek',
        'dragonage',
        'buffy',
        'jojo',
        'pokemon',
        'danganronpa',
        'glee',
        'fire_emblem',
        'hannibal',
        'dcu',
        'titan',
        'harrypotter',
        'percy_jackson',
        'tolkien',
        'naruto',
        'teenwolf',
        'song_ice_fire',
        'shadowhunters',
        'walking_dead'
    ]
    dataset_name = 'complete_en_1k-5k'
    base_dirpath = '/data/fanfiction_ao3'
    output_dirpath = '/data/fanfiction_lgbtq/'

    dataset = CharacterFeaturesDataset(fandoms, base_dirpath, dataset_name, 
        output_dirpath)
    dataset.transform()
    dataset.save()

if __name__ == '__main__':
    main()
