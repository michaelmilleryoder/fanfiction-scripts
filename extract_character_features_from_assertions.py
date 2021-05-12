import os
import re
import json
from multiprocessing import Pool
import pdb

from tqdm import tqdm
import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en', disable=['ner'])
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)


class AssertionProcessor:
    """ Process assertions in a fandom """

    def __init__(self, pipeline_output_dirpath):
        self.assertions_dirpath = os.path.join(pipeline_output_dirpath, 
            'assertion_extraction')
        self.coref_dirpath = os.path.join(pipeline_output_dirpath, 
            'char_coref')
        self.output_dirpath = os.path.join(pipeline_output_dirpath, 
            'char_features')
        if not os.path.exists(self.output_dirpath):
            os.mkdir(self.output_dirpath)
        self.stops = ['was', 'were', 'to', 'for', 'in', 'on', 'by', 'has', 
            'had', 'been', 'be', "'re",'’re', '’ll', "'ll", "'s", '’s', '’ve', "'ve",
             "'m", '’m', "n't", 'n’t', 'at', 'of', 'a', 'an', 'i', 'you']

    def load_character_assertions(self, fandom_fname):
        """ Load character assertions for a fic """
        fic_assertions_path = os.path.join(self.assertions_dirpath, 
            fandom_fname + '.json')
        with open(fic_assertions_path) as f:
            char_assertions = json.load(f)
        return char_assertions

    def load_coref(self, fandom_fname):
        """ Load coref info for a fic, to identify character mentions """
        coref_fpath = os.path.join(self.coref_dirpath, fandom_fname + '.json')
        with open(coref_fpath) as f:
            coref = json.load(f)
        return coref

    def extract_features(self, assertion, char, coref):
        """ Extract desired features from a fic's assertion """

        text = assertion['text']
        
        # postag and parse
        annotated = nlp(text)
    
        # Get character mention locations
        cluster_matches = [clus for clus in coref['clusters'] if clus.get(
            'name', '') == char]
        if len(cluster_matches) == 0:
            return []
        cluster = cluster_matches[0]
        mention_inds = [list(range(m['position'][0], m['position'][1])) \
            for m in cluster['mentions']] # token IDs of character in the text
        mention_inds = [i for el in mention_inds for i in el]

        # Verbs where character was the subject
        offset = assertion['position'][0]
        verbs_subj = [tok.head.text.lower() for tok in annotated if tok.i + offset \
            in mention_inds and (tok.dep_=='nsubj' or tok.dep_=='agent')]

        # Verbs where character was the object
        verbs_obj = [tok.head.text.lower() for tok in annotated \
            if tok.i + offset in mention_inds and \
            (tok.dep_=='dobj' or tok.dep_=='nsubjpass' or \
            tok.dep_=='dative' or tok.dep_=='pobj')]

        # Adjectives that describe the character
        adjs = [tok.text.lower() for tok in annotated if tok.head.i + offset in \
            mention_inds and (tok.dep_=='amod' or tok.dep_=='appos' or \
            tok.dep_=='nsubj' or tok.dep_=='nmod')] \
            + [tok.text.lower() for tok in annotated if tok.dep_=='attr' and \
                (tok.head.text=='is' or tok.head.text=='was') and \
               any([c.i + offset in mention_inds for c in tok.head.children])]

        # Remove stopwords
        final_list = [w for w in verbs_subj + verbs_obj + adjs if not w in self.stops]
        return final_list

    def process_fic(self, fname):
        """ Process assertions from an individual fic """
        fandom_fname = fname.split('.')[0]
        outpath = os.path.join(self.output_dirpath, f'{fandom_fname}.json')
        char_assertions = self.load_character_assertions(fandom_fname)
        coref = self.load_coref(fandom_fname)
        char_features = {} 

        # Check if already processed
        if os.path.exists(outpath):
            return

        # Process
        for char in char_assertions:
            if not char in char_features:
                char_features[char] = []
            for assertion in char_assertions[char]:
                assertion_features = self.extract_features(assertion, char, coref)
                char_features[char].extend(assertion_features)

        # Save out
        with open(outpath, 'w') as f:
            json.dump(char_features, f)

    def process(self):
        """ Process assertions to get character actions and attributes """

        fnames = sorted(os.listdir(self.assertions_dirpath))

        # Process assertions
        with Pool(30) as p:
            list(tqdm(p.imap(self.process_fic, fnames), total=len(fnames), ncols=70))

        # for debugging
        #list(map(self.process_fic, tqdm(sorted(os.listdir(self.assertions_dirpath)),
        #    ncols=70)))


def main():
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

    for fandom in fandoms:
        pipeline_output_path = f'/data/fanfiction_ao3/{fandom}/{dataset_name}/output/'
        if not os.path.exists(os.path.join(pipeline_output_path, 
            'assertion_extraction')):
            print(f'Skipping {fandom}')
            continue
        print(f'Processing {fandom}')
        processor = AssertionProcessor(pipeline_output_path)
        processor.process()


if __name__ == '__main__': main()
