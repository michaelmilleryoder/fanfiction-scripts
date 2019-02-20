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
    #'song_ice_fire',
]

data_dirpath_template = '/usr2/scratch/fanfic/ao3_{}_text/stories'
metadata_fpath = '/usr2/scratch/fanfic/ao3_{}_text/stories.csv'
#fic_metadata = '/usr0/home/prashang/DirectedStudy/ACL_preprocessing/stories_kudos50.csv'
output_dirpath = '/usr2/mamille2/fanfiction-project/data/ao3/{0}/fics'

#nlp = spacy.load('en', disable=['ner'])
#nlp = spacy.load('en', disable=['parser', 'ner']) # 1 para/line
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner']) # 1 para/line

for fandom in fandoms:
    print(fandom)
    fandom_dirpath = data_dirpath_template.format(fandom)
    output_fpath = os.path.join(output_dirpath.format(fandom), '{0}_tokenized_paras.txt')
    
    if not os.path.exists(output_dirpath.format(fandom)):
        os.makedirs(output_dirpath.format(fandom))

    # Get fic text from all chapters, concatenate and save
    fic_metadata = pd.read_csv(metadata_fpath.format(fandom))
    for fid, chapter_count in tqdm(zip(fic_metadata['fic_id'], fic_metadata['chapter_count']), total=len(fic_metadata)):
        chap_fnames = []
        fic_paras = []
        tqdm.write(str(chapter_count))
        for i in range(chapter_count):
            chap_fnames.append(f"{fid}_{i+1:04}.csv")

        #for fname in tqdm(os.listdir(fandom_dirpath)):
        with open(output_fpath.format(fid), 'w') as f:
            
            #fic_text = ''
            for fname in chap_fnames:
                for para in pd.read_csv(os.path.join(fandom_dirpath, fname))['text'].tolist():
                    if isinstance(para, str):
                        #fic_text += ' ' + para

                        #doc = nlp(fic_text)
                        toks_str = ' '.join([tok.text for tok in nlp.tokenizer(para.lower())])
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
          
