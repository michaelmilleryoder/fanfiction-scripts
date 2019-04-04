import os
from tqdm import tqdm
#from nltk import sent_tokenize
import pandas as pd
import pdb
from multiprocessing import Pool
import spacy
import argparse
import re
#from nltk import word_tokenize
#from nltk.tokenize import sent_tokenize


# Settings
parser = argparse.ArgumentParser()
parser.add_argument('tokenization', nargs='?', help='Sentence or paragraph written per line. Enter "sent" or "para"')
parser.add_argument('n_jobs', nargs='?', help='Number of threads')
args = parser.parse_args()

tokenization = args.tokenization # 'sent' or 'para'
n_jobs = int(args.n_jobs)


# I/0
data_dirpath = '/usr2/mamille2/coca/fiction'
output_dirpath = '/usr2/mamille2/coca/fiction-preprocessed'

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

if tokenization == 'sent':
    nlp.add_pipe(nlp.create_pipe('sentencizer'))


def preprocess(fname, lines):
    """ Remove unwanted characters, break into segments/line.
        COCA is already tokenized.
     """
    
    output_fpath = os.path.join(output_dirpath, fname)

    with open(output_fpath, 'w') as f:

        for text in lines:

            # Remove page markers
            text = text.replace('@ @ @ @ @ @ @ @ @ @ ', '')

            if fname.startswith('w'):
                # Remove story headers
                text = re.sub(r'^##\d+ ', '', text)

                # Split on paragraph markers
                paras = text.split('<p>')

            else:
                # Remove story headers
                text = re.sub(r'^@@\d+ ', '', text)

                # Split on paragraph markers
                paras = text.split('#')

            if tokenization == 'para':
                # 1 para/line
                for para in paras:
                    f.write(f"{para}\n")

            elif tokenization == 'sent':
                pass
                
                # nltk
                #for sent in sent_tokenize(chap_text):
                #    toks_str = ' '.join([tok for tok in word_tokenize(sent)])
                #    f.write(f"{toks_str}\n")

                # spacy
#                for sent in doc.sents:
#                    toks_str = ' '.join([tok.text for tok in sent])
#                    f.write(f"{toks_str}\n")

        # 1 sentence/line
#            for sent in doc.sents:
#                toks_str = ' '.join([tok.text for tok in sent])
#                fil.write(f"{toks_str}\n")


def main():
    print("Processing fiction texts...")
    print(f'Will save to {output_dirpath}')
    
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    fnames = [f for f in sorted(os.listdir(data_dirpath)) if not f.startswith('.')]

    for fname in tqdm(fnames):
        tqdm.write(fname)
        with open(os.path.join(data_dirpath, fname)) as f:
            text = f.read().splitlines()
            preprocess(fname, text)

    #with Pool(n_jobs) as p:
    #    list(tqdm(p.imap(preprocess_fic, list(zip(fic_metadata['fic_id'], fic_metadata['chapter_count']))), total=len(fic_metadata)))
              
if __name__ == '__main__':
    main()
