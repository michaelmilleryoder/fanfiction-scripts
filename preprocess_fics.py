""" A combination of utilities for preprocessing fanfiction scraped from AO3.
    @author Michael Miller Yoder <yoder@cs.cmu.edu>
    @date 2020
"""
import argparse
import os
from multiprocessing import Pool
import spacy
import pdb
import pandas as pd
from tqdm import tqdm

from filter_fics import get_fic2chapter, copy_fics as copy_stories

print("Loading tokenizer...")
NLP = spacy.load('en')

class Preprocessor():

    def __init__(self, scraped_dirpath, out_dirpath, update):
        self.scraped_dirpath = scraped_dirpath
        self.scraped_fic_dirpath = os.path.join(self.scraped_dirpath, 'stories')
        self.out_dirpath = out_dirpath
        self.fics_out_dirpath = os.path.join(self.out_dirpath, 'fics')
        self.update = update
        self.metadata = None

    def copy_metadata(self, update=False):
        """ Copy fic metadata to output directory """
        print("Copying metadata...")
        metadata_outpath = os.path.join(self.out_dirpath, 'metadata.csv')
        if self.metadata is None:
            self.load_metadata()
        if update:
            # Load existing metadata
            existing_metadata = pd.read_csv(metadata_outpath)
            self.metadata = pd.concat([existing_metadata, self.metadata]).drop_duplicates('fic_id')
        # Save out metadata
        self.metadata.to_csv(metadata_outpath)

    def copy_fics(self):
        print("Copying fics...")
        fic2chapter = get_fic2chapter(self.scraped_fic_dirpath)
        if self.metadata is None:
            self.load_metadata()
        fics_with_no_data = copy_stories(self.metadata['fic_id'].tolist(), fic2chapter, self.scraped_fic_dirpath, self.out_dirpath, num_cores=30)
        pdb.set_trace()
        self.metadata = self.metadata[~self.metadata['fic_id'].isin(fics_with_no_data)] # remove metadata for fics that don't have data

    def tokenize_fics(self, num_cores=-1):
        """ Tokenize fics in output (copied) directory
            Args:
                num_cores: # of cores (-1 for none)
        """
        
        print("Tokenizing fics...")
        fic_fpaths = [os.path.join(self.fics_out_dirpath, fname) for fname in os.listdir(self.fics_out_dirpath)]

        if num_cores > 0:
            with Pool(num_cores) as p:
                list(tqdm(p.imap(tokenize_fic, fic_fpaths), total=len(fic_fpaths)))

        else:
            # without multiprocessing for debugging
            list(map(tokenize_fic, fic_fpaths))

    def load_metadata(self):
        print("\tLoading metadata...")
        self.metadata = pd.read_csv(os.path.join(self.scraped_dirpath, 'stories.csv'))
        # Resolve bytestring issue (should be resolved with newer scrapes)
        self.metadata = self.metadata.applymap(remove_bytestring)

    def combine_chapters(self):
        """ Combine chapters into one big CSV for a work/story """

    def preprocess(self):
        self.copy_fics() # combines chapters into single fic CSVs
        self.copy_metadata(update=self.update)
        self.tokenize_fics(num_cores=15)


def remove_bytestring(value):
    new_value = value
    if isinstance(value, str):
        if value.startswith("b'") or value.startswith('b"'):
            new_value = value[2:-1]
    return new_value


def tokenize(text):
    return ' '.join([tok.text for tok in NLP.tokenizer(text)])


def tokenize_fic(fpath, force=False):
    """ Tokenize a fic.
        Args:
            fpath: Path to a CSV file with fic data. The column "text" will be tokenized and a column "text_tokenized" added.
            force: Whether to force new tokenization if the CSV already contains a "text_tokenized" column (default=False)
    """
    
    fic = pd.read_csv(fpath)
    if 'text_tokenized' in fic.columns:
        return
    fic['text_tokenized'] = fic['text'].map(tokenize)
    fic.to_csv(fpath, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scraped_dirpath', nargs='?',
            help='Path to the directory of scraped AO3 fics. (See https://github.com/michaelmilleryoder/AO3Scraper)')
    parser.add_argument('out_dirpath', nargs='?',
            help='If combining fic IDs, path to the directory where the fic CSVs are')
    parser.add_argument('--update-existing', dest='update', action='store_true',
            help='Set this flag if the out_dirpath is an existing directory and you wish to update the fics already stored in it.')

    return parser.parse_args()


def main():

    args = get_args()
    processor = Preprocessor(args.scraped_dirpath, args.out_dirpath, args.update)
    processor.preprocess()


if __name__ == '__main__':
    main()
