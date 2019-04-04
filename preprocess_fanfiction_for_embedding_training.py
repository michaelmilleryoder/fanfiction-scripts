import os
from tqdm import tqdm
#from nltk import sent_tokenize
import pandas as pd
import pdb
from multiprocessing import Pool
import spacy
import argparse
#from nltk import word_tokenize
#from nltk.tokenize import sent_tokenize


# Settings
parser = argparse.ArgumentParser()
parser.add_argument('fandom', nargs='?', help='Name of fandom')
parser.add_argument('tokenization', nargs='?', help='Sentence or paragraph written per line')
parser.add_argument('n_jobs', nargs='?', help='Number of threads')
args = parser.parse_args()

fandom = args.fandom #'song_ice_fire',
tokenization = args.tokenization # 'sent' or 'para'
n_jobs = int(args.n_jobs)


# I/0

data_dirpath_template = '/usr2/scratch/fanfic/ao3_{}_text/stories'
fandom_dirpath = data_dirpath_template.format(fandom)
metadata_fpath = '/usr2/scratch/fanfic/ao3_{}_text/stories.csv'
#fic_metadata = '/usr0/home/prashang/DirectedStudy/ACL_preprocessing/stories_kudos50.csv'
output_dirpath = '/usr2/mamille2/fanfiction-project/data/ao3/{0}/fics_{1}s'
fandom_out_dirpath = output_dirpath.format(fandom, tokenization)

# Global (should incorporate into map)
#nlp = spacy.load('en', disable=['ner'])
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

if tokenization == 'sent':
    nlp.add_pipe(nlp.create_pipe('sentencizer'))



def preprocess_fic(fic_info):
            fic_id = fic_info[0]
            chapter_count = fic_info[1]
            output_fpath = os.path.join(fandom_out_dirpath, f'{fic_id}.txt')
            chap_fnames = []
            fic_paras = []
            for i in range(chapter_count):
                chap_fnames.append(f"{fic_id}_{i+1:04}.csv")

            #for fname in tqdm(os.listdir(fandom_dirpath)):
            with open(output_fpath, 'w') as f:
                
                #fic_text = ''
                for fname in chap_fnames:
                    chap_text = ''
                    for para in pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist():
                        if isinstance(para, str):
                            #fic_text += ' ' + para
                            chap_text += ' ' + para

                    #doc = nlp(fic_text)
                    doc = nlp(chap_text)

                    if tokenization == 'para':
                        toks_str = ' '.join([tok.text for tok in nlp.tokenizer(para.lower())])
                    
                    elif tokenization == 'sent':
                        
                        # nltk
                        #for sent in sent_tokenize(chap_text):
                        #    toks_str = ' '.join([tok for tok in word_tokenize(sent)])
                        #    f.write(f"{toks_str}\n")

                        # spacy
                        for sent in doc.sents:
                            toks_str = ' '.join([tok.text for tok in sent])
                            f.write(f"{toks_str}\n")

                    # 1 paragraph/line
                    #paras = [p for p in pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist() if isinstance(p, str)]
                    #fic_paras += paras

                    # 1 sentence/line
                    #paras = ' '.join([p for p in pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist() if isinstance(p, str)])
                    #doc = nlp(paras)


                # 1 para/line
    #            for para in fic_paras:
    #                doc = nlp(para)
    #                toks_str = ' '.join([tok.text for tok in doc])
    #                f.write(f"{toks_str}\n")

                # 1 sentence/line
    #            for sent in doc.sents:
    #                toks_str = ' '.join([tok.text for tok in sent])
    #                fil.write(f"{toks_str}\n")

def main():
    print(fandom)
    print(f'Will save to {fandom_out_dirpath}')
    
    if not os.path.exists(fandom_out_dirpath):
        os.makedirs(fandom_out_dirpath)

    # Get fic text from all chapters, concatenate and save
    fic_metadata = pd.read_csv(metadata_fpath.format(fandom))
    with Pool(n_jobs) as p:
        list(tqdm(p.imap(preprocess_fic, list(zip(fic_metadata['fic_id'], fic_metadata['chapter_count']))), total=len(fic_metadata)))
        #for fid, chapter_count in tqdm(zip(fic_metadata['fic_id'], fic_metadata['chapter_count']), total=len(fic_metadata)):
              
if __name__ == '__main__':
    main()
