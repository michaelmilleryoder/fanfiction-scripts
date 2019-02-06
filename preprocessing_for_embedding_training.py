import os
import spacy
from tqdm import tqdm
#from nltk import sent_tokenize
import pandas as pd
import pdb


# Load fanfiction data
fandoms = [
    'harrypotter',
    #'academia',
#    'allmarvel',
#    'detroit',
    #'friends',
    #'supernatural',
]

data_dirpath_template = '/usr2/scratch/fanfic/ao3_{}_text/stories'
fic_metadata = '/usr0/home/prashang/DirectedStudy/ACL_preprocessing/stories_kudos50.csv'
output_dirpath = '/usr2/mamille2/fanfiction-project/data/ao3/{0}'

nlp = spacy.load('en', disable=['ner'])

for fandom in fandoms:
    print(fandom)
    fandom_dirpath = data_dirpath_template.format(fandom)
    output_fpath = os.path.join(output_dirpath.format(fandom), '{0}_sentences.txt')
    
    if not os.path.exists(output_dirpath.format(fandom)):
        os.makedirs(output_dirpath.format(fandom))

    # Get fic fnames
    fic_fnames = []
    fic_metadata = pd.read_csv(fic_metadata)
    for fid, chapter_count in zip(fic_metadata['fic_id'], fic_metadata['chapter_count']):
        for i in range(chapter_count):
            fic_fnames.append(f"{fid}_{i+1:04}.csv")

    #for fname in tqdm(os.listdir(fandom_dirpath)):
    for fname in tqdm(fic_fnames):
        #paras = pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist()
        paras = ' '.join([p for p in pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist() if isinstance(p, str)])

        # spaCy sentence tokenizer
        doc = nlp(paras)

        with open(output_fpath.format(fname[:-4]), 'w') as fil:
            for sent in doc.sents:
                toks_str = ' '.join([tok.text for tok in sent])
                fil.write(f"{toks_str}\n")
          
            #for para in paras: # might be faster to join
            #    if not isinstance(para, str):
            #        continue
            #    doc = nlp(para)

            #    for sent in doc.sents:
            #        toks_str = ' '.join([tok.text for tok in sent])
            #        pdb.set_trace()
            #        fil.write(f"{toks_str}\n")
