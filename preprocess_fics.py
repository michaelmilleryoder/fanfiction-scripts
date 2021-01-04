""" Preprocess fanfiction scraped from AO3.
    @author Michael Miller Yoder <yoder@cs.cmu.edu>
    @date 2020
"""
import argparse
import os
from multiprocessing import Pool
import datetime
import pdb
import shutil
import itertools

import spacy
import pandas as pd
from tqdm import tqdm

from filter_fics import get_fic2chapter, copy_fics as copy_stories
import preprocess_utils as utils

print("Loading tokenizer...")
NLP = spacy.load('en')


class Preprocessor():

    def __init__(self, scraped_dirpath, out_dirpath, update, merge_chapters, num_copy_workers, num_tok_workers):
        self.scraped_dirpath = scraped_dirpath
        self.scraped_fic_dirpath = os.path.join(self.scraped_dirpath, 'stories')
        self.out_dirpath = out_dirpath
        self.fics_out_dirpath = os.path.join(self.out_dirpath, 'fics')
        self.update = update
        self.merge_chapters = merge_chapters
        self.metadata = None
        self.num_copy_workers = num_copy_workers
        self.num_tok_workers = num_tok_workers

    def copy_metadata(self, update=False):
        """ Copy fic metadata to output directory """
        print("Copying metadata...")
        metadata_outpath = os.path.join(self.out_dirpath, 'metadata.csv')
        if self.metadata is None:
            self.load_scraped_metadata()
        if update:
            # Load existing metadata
            existing_metadata = utils.load_metadata_file(metadata_outpath)
            self.metadata = pd.concat([existing_metadata, self.metadata]).drop_duplicates('fic_id').loc[:, self.metadata.columns] # remove any extra columns
        # Save out metadata
        self.metadata.to_csv(metadata_outpath, index=False)

    def copy_fics(self, merge_chapters=False, num_cores=1):
        """ Copy selected fics to output directory.
            Args:
                merge_chapters: If True, merge chapter files in the input to output.
                        Set to False if the input is already in whole stories.
                num_cores: number of cores to copy files. Default is 1.
        """
        print("Copying fics...")
        if self.metadata is None:
            self.load_scraped_metadata()
        if merge_chapters:
            fic2chapter = get_fic2chapter(self.scraped_fic_dirpath)
            fics_with_no_data = copy_stories(self.metadata['fic_id'].tolist(), fic2chapter, self.scraped_fic_dirpath, self.out_dirpath, num_cores=num_cores)
        else:
            fics_with_no_data = utils.just_copy_fics(self.metadata['fic_id'].tolist(), self.scraped_fic_dirpath, self.fics_out_dirpath, num_cores=num_cores)
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
                list(tqdm(p.imap(utils.tokenize_fic, fic_fpaths), total=len(fic_fpaths), ncols=70))

        else:
            # without multiprocessing for debugging
            #list(map(utils.tokenize_fic, tqdm(fic_fpaths, ncols=70)))
            utils.tokenize_fic(os.path.join(self.fics_out_dirpath, '22923991.csv'), force=True)

    def load_scraped_metadata(self):
        print("\tLoading metadata...")
        self.metadata = utils.load_metadata_file(os.path.join(self.scraped_dirpath, 'stories.csv'))

    def preprocess(self):
        self.copy_fics(merge_chapters=self.merge_chapters, num_cores=self.num_copy_workers)
        self.copy_metadata(update=self.update)
        self.tokenize_fics(num_cores=self.num_tok_workers)
        self.align_metadata_fics()

    def align_metadata_fics(self):
        """ Check if all copied fics have metadata and vice versa.
            Remove any that are not present (due to errors, likely)
        """
        # Load fics
        saved_fics = [int(fname[:-4]) for fname in os.listdir(self.fics_out_dirpath)]
        print(f'Number of copied/saved fics: {len(saved_fics)}')

        # Compare with metadata
        metadata_outpath = os.path.join(self.out_dirpath, 'metadata.csv')
        if self.metadata is None:
            self.metadata = utils.load_metadata_file(metadata_outpath)
        print(f'Number of fics with metadata: {len(self.metadata)}')
        
        # Any mismatches
        metadata_no_fic = set(self.metadata['fic_id'].unique()) - set(saved_fics)
        fic_no_metadata = set(saved_fics) - set(self.metadata['fic_id'].unique())
        print(f'Number of fics with metadata but without text: {len(metadata_no_fic)}')
        print(f'Number of fics with text but no metadata: {len(fic_no_metadata)}')

        # Remove any metadata that doesn't have text
        if len(metadata_no_fic) > 0:
            self.metadata = self.metadata[~self.metadata['fic_id'].isin(metadata_no_fic)]
            # Move previous metadata
            backup_metadata_outpath = metadata_outpath + f'.orig-{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")}'
            os.rename(metadata_outpath, backup_metadata_outpath)
            self.metadata.to_csv(metadata_outpath, index=False)
            print("Removed metadata for fics with no data")

        # Update info.txt
        with open(os.path.join(self.out_dirpath, 'info.txt'), 'a') as f:
            f.write(f'\nUpdated {datetime.datetime.now().strftime("%Y-%m-%d")}\n')
            f.write(f'\tTotal number of fics: {len(self.metadata)}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scraped_dirpath', nargs='?',
            help='Path to the directory of scraped AO3 fics. (See https://github.com/michaelmilleryoder/AO3Scraper)')
    parser.add_argument('out_dirpath', nargs='?',
            help='If combining fic IDs, path to the directory where the fic CSVs are')
    parser.add_argument('--update-existing', dest='update', action='store_true',
            help='Set this flag if the out_dirpath is an existing directory and you wish to update the fics already stored in it.')
    parser.add_argument('--merge-chapters', dest='merge_chapters', action='store_true', default=False,
            help='Set this flag if the scraped data is in chapters instead of whole fic CSVs.')
    parser.add_argument('--num-tok-workers', dest='num_tok_workers', nargs='?', type=int, default=1,
            help='Number of processes to run the tokenization with.')
    parser.add_argument('--num-copy-workers', dest='num_copy_workers', nargs='?', type=int, default=1,
            help='Number of processes to copy files.')
    return parser.parse_args()


def main():
    args = get_args()
    processor = Preprocessor(args.scraped_dirpath, args.out_dirpath, args.update, args.merge_chapters, args.num_copy_workers, args.num_tok_workers)
    processor.preprocess()


if __name__ == '__main__':
    main()
