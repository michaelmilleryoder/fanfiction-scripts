from gensim.models import KeyedVectors, FastText 
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
from tqdm import tqdm as tqdm
import pickle
import pdb
import argparse
import re
from multiprocessing import Pool, Manager, Process
import itertools
from nltk.tokenize import sent_tokenize
import nltk
import json


def pairing_present(text, pairing):

    if re.search(r'(\b|_){}(\b|_)'.format(pairing[0]), text) and re.search(r'(\b|_){}(\b|_)'.format(pairing[1]), text):
        return True
    else:
        return False

def extract_character_quote(params):
    """ Extract character quotes from a fic """

    fname = params[0]
    pairings = params[1]
    input_fpath = params[2]
    context_windows = params[3]
    fic_pairing_paragraph_threshold = params[4]
    chars = params[5]
    context = params[6]

    fic_id = int(fname.split('.')[0])

    fic_contexts = []
    
    with open(os.path.join(input_fpath, fname)) as f:

        quotes = json.load(f)

    # Load tmp file
    token_fpath = '/usr0/home/mamille2/fanfiction-project/fanfiction-nlp/quote_attribution/tmp/{}/token_tmp.txt'
    tokens = pd.read_csv(token_fpath.format(fic_id), sep='\t', quoting=csv.QUOTE_NONE)

    speakers = set([s['speaker'].lower() for s in quotes])
    
    for pairing in pairings:
        if any([pairing[0] in s for s in speakers]) and any([pairing[1] in s for s in speakers]):

            # Grab all paragraphs from that speaker, see if replying to the other
            for c in pairing:
                speaker = [s for s in speakers if c in s]
                other_char = [other for other in pairing if other!=c][0]
                for s in speaker: # could have multiple corresponding speakers
                    speaker_quotes = [q for q in quotes if q['speaker']==s]
                    speaker_replies = [q for q in speaker_quotes if q['replyto'] != -1]

                    # Check reply to what quote
                    for quote in speaker_replies:
                        replied_para = quote['replyto']
                        replied_to = [q for q in quotes if q['paragraph']==replied_para][0]
                        if replied_to['speaker'] == other_char:
                            para_quotes = ' '.join([t['quote'].lower() for t in quote['quotes']])
                            replied_to_quotes = ' '.join([t['quote'].lower() for t in replied_to['quotes']])
                            pairing_quote = ' '.join([para_quotes, replied_to_quotes])
                            fic_contexts.extend(pairing_quote.split())

                    # Check if other is in quote
                    for quote in speaker_quotes:
                        added_quote = False
                        para_quotes = ' '.join([t['quote'].lower() for t in quote['quotes']])
                        if other_char in para_quotes:
                            fic_contexts.extend([tok for tok in para_quotes.split() if not tok.startswith('ccc_')])
                            break
                            
                        # Check if other is in following or previous paragraph
                        para_id = quote['paragraph']
                        prev_para = ' '.join(tokens.loc[tokens['paragraphId']==para_id-1,
                                                'originalWord'].dropna().str.lower().tolist())
                        if other_char in prev_para:
                            fic_contexts.extend([tok for tok in para_quotes.split() if not tok.startswith('ccc_')])
                            break
        
                        next_para = ' '.join(tokens.loc[tokens['paragraphId']==para_id+1,
                                                'originalWord'].dropna().str.lower().tolist())
                        if other_char in next_para:
                            fic_contexts.extend([tok for tok in para_quotes.split() if not tok.startswith('ccc_')])
        
        char_contexts_fanfic[pairing][fic_id] = fic_contexts


def extract_character_quote_restrictive(params):
    """ Extract character quotes from a fic """

    fname = params[0]
    pairings = params[1]
    input_fpath = params[2]
    context_windows = params[3]
    fic_pairing_paragraph_threshold = params[4]
    chars = params[5]
    context = params[6]

    fic_id = int(fname.split('.')[0])

    fic_contexts = []
    
    with open(os.path.join(input_fpath, fname)) as f:

        quotes = json.load(f)

    speakers = set([s['speaker'].lower() for s in quotes])
    
    for pairing in pairings:
        if any([pairing[0] in s for s in speakers]) and any([pairing[1] in s for s in speakers]):

            # Grab all paragraphs from that speaker, see if replying to the other
            for c in pairing:
                speaker = [s for s in speakers if c in s]
                for s in speaker: # could have multiple corresponding speakers
                    speaker_quotes = [q for q in quotes if q['speaker']==s]
                    speaker_replies = [q for q in speaker_quotes if q['replyto'] != -1]

                    # Check reply to what quote
                    for quote in speaker_replies:
                        replied_para = quote['replyto']
                        replied_to = [q for q in quotes if q['paragraph']==replied_para][0]
                        other_char = [other for other in pairing if other!=c][0]
                        if replied_to['speaker'] == other_char:
                            para_quotes = ' '.join([t['quote'].lower() for t in quote['quotes']])
                            replied_to_quotes = ' '.join([t['quote'].lower() for t in replied_to['quotes']])
                            pairing_quote = ' '.join([para_quotes, replied_to_quotes])
                            fic_contexts.extend(pairing_quote.split())
        
        char_contexts_fanfic[pairing][fic_id] = fic_contexts


def extract_character_assertion(params):
    """ Extract character assertions from a fic """

    fname = params[0]
    pairings = params[1]
    input_fpath = params[2]
    context_windows = params[3]
    fic_pairing_paragraph_threshold = params[4]
    chars = params[5]
    context = params[6]

    fic_id = int(fname.split('.')[0])
    
    with open(os.path.join(input_fpath, fname)) as f:

        assertions = json.load(f)

    matches = [charname for charname in assertions if any([c in charname.lower() for c in chars])]

    if len(matches) < 2:
        return

    possible_pairings = [pairing for pairing in pairings if \
                            any([pairing[0] in m.lower() for m in matches]) and \
                            any([pairing[1] in m.lower() for m in matches])]

    for pairing in possible_pairings:

        fic_pairing_count = 0
        if context == 'all':
            fic_contexts = []
        elif context == 'local':
            fic_contexts = {c: {w: [] for w in context_windows} for c in pairing}
        matching_chars = [charname for charname in assertions if any([c in charname.lower() for c in pairing])]

        all_assertions = [assertions[k] for k in matching_chars]
        all_assertions = [assertion for group in all_assertions for assertion in group]
        all_assertions = [assertion.lower().split() for assertion in all_assertions]
        for assertion in all_assertions:

            # Determine whether both characters are in the assertion
            if pairing_present(' '.join(assertion), pairing):
                fic_pairing_count += 1

                if context == 'all':
                    # Remove character labeling
                    #toks = [tok for tok in assertion.lower().split() ]
                    fic_contexts.extend([w for w in assertion if not w in pairing])

                elif context == 'local':
                    for c in pairing:
                        match = [charname for charname in matching_chars if c in charname.lower()] # find matching charnames
                        for charname in match:
                            for idx in [i for i,token in enumerate(assertion) if token==charname.lower()]:
                                for context_window in context_windows:
                                    
                                    assertion_nonames = [tok for tok in assertion if not tok.startswith('($_')]
                                    toks_before = assertion_nonames[max(0, idx-context_window) : idx]
                                    fic_contexts[c][context_window] += toks_before
                                    toks_after = assertion_nonames[idx+1 : min(idx+1+context_window, len(assertion))]
                                    fic_contexts[c][context_window] += toks_after
        
        if fic_pairing_count >= fic_pairing_paragraph_threshold:
            for c in pairing:
                for context_window in context_windows:
                    char_contexts_fanfic[pairing][c][context_window][fic_id] = fic_contexts[c][context_window]


def extract_character_paragraph(params):
    fname = params[0]
    pairings = params[1]
    input_fpath = params[2]
    fic_pairing_paragraph_threshold = params[4]
    save_paras_fpath = params[5]

    fic_id = int(fname.split('_')[0])
    
    with open(os.path.join(input_fpath, fname)) as f:
        paras = [p.split() for p in f.read().splitlines()]
        paras_present = []

        #for c in chars:
        for pairing in pairings:

            fic_pairing_count = 0
            fic_contexts = []
                
            for para in paras:

                # Determine whether both characters are in the paragraph
                if pairing_present(' '.join(para), pairing):

                    paras_present.append(' '.join(para))
                    fic_contexts.extend([w for w in para if not w in pairing])
                    fic_pairing_count += 1

            if fic_pairing_count >= fic_pairing_paragraph_threshold:

                if save_paras_fpath:

                    with open(save_paras_fpath.format('_'.join(pairing), fic_id), 'a') as f:
                        # Sentence-tokenize paragraphs
                        for para in paras_present:
                            sents = sent_tokenize(para)
                            for sent in sents:
                                f.write(f"{sent}\n")
                            f.write('\n')

                char_contexts_fanfic[pairing][fic_id] = fic_contexts


def extract_character_local_context(params):
    fname = params[0]
    pairings = params[1]
    input_fpath = params[2]
    context_windows = params[3]
    fic_pairing_paragraph_threshold = params[4]
    save_paras_fpath = params[5]

    fic_id = int(fname.split('_')[0])
    
    with open(os.path.join(input_fpath, fname)) as f:
        paras = [p.split() for p in f.read().splitlines()]
        paras_present = []

        #for c in chars:
        for pairing in pairings:

            fic_pairing_count = 0
            fic_contexts = {w: [] for w in context_windows}
                
            for para in paras:

                # Determine whether both characters are in the paragraph
                if pairing_present(' '.join(para), pairing):

                    paras_present.append(' '.join(para))

                    fic_pairing_count += 1

                    for c in pairing:
                        for idx in [i for i,token in enumerate(para) if token==c]:
                            for context_window in context_windows:
                                
                                fic_contexts[context_window] += para[max(0, idx-context_window) : idx] # before
                                fic_contexts[context_window] += para[idx+1 : min(idx+1+context_window, len(para))] # after
        
            if fic_pairing_count >= fic_pairing_paragraph_threshold:

                if save_paras_fpath:

                    with open(save_paras_fpath.format('_'.join(pairing), fic_id), 'a') as f:
                        # Sentence-tokenize paragraphs
                        for para in paras_present:
                            sents = sent_tokenize(para)
                            for sent in sents:
                                f.write(f"{sent}\n")
                            f.write('\n')

                for c in pairing:
                    for context_window in context_windows:
                        char_contexts_fanfic[pairing][c][context_window][fic_id] = fic_contexts[context_window]


def main():

    # Settings
    chars = [
        'harry',
        'hermione',
        'ron',
        'ginny',
        'draco',
#        'neville',
#        'luna',
#        'remus',
#        'sirius',
#        'severus',
#        'james',
#        'lily',
    ]

    pairings = [
        ('draco', 'harry'),
        ('hermione', 'ron'),
        ('ginny', 'harry'),
        ('draco', 'hermione'),
        ('harry', 'hermione'),
        ('harry', 'ron'),
    ]

    vector_combination = 'ngrams1' # {'add', 'context_only', 'ngrams1'}
    input_type = 'quote' # {'assertion', 'quote', 'paragraph'}
    context = 'all' # {'local', 'all'}
    extract_contexts = True
    nonames = True # remove names and pronouns from consideration
    fic_pairing_paragraph_threshold = 5
    #context_windows = [5, 10, 25, 50] # before and after, so total window is this value * 2
    context_windows = [10] # before and after, so total window is this value * 2
    mt_align = False
    #save_paras_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/pairings/{}/{}_paras.txt' # None for not saving this
    save_paras_fpath = None

    if nonames:

        with open('harrypotter_characters.txt') as f:
            char_names = [c.lower() for c in f.read().splitlines()]
            char_names = list(set([name for c in char_names for name in c.split()]))

        pronouns = [
                    'she', 'they', 'he', 'we', 'i',
                    'her', 'their', 'him', 'us', 'me',
                    'hers', 'theirs', 'his', 'our', 'my', 'ours', 'mine',
                    ]

        stopwords = nltk.corpus.stopwords.words('english') + char_names + pronouns

    # TSNE settings
    run_tsne = False
    tsne_fic_ids_path = None # if None, then run TSNE on all fics
    #tsne_fic_ids_path = '/usr0/home/mamille2/erebor/fanfiction-project/data/qian_sample1_fic_ids.csv' # if None, then run TSNE on all fics
    tsne_char_embs_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_chars_lc10_context.nptxt'
    tsne_char_labels_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_chars_lc10_context_labels.txt'
    tsne_fname_labels_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_fnames_lc10_context.txt'
    #plot_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_chars_lc10_context_plot.png'

    # I/O
    fanfic_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/fanfic.harry_potter.all.lower.model.vec'
    #fanfic_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/background.lower.model.vec'
    #canon_mt_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/canon.harry_potter.lower.model.vec'
    fanfic2bg_path = '/usr0/home/qinlans/ACL_2019/transformation_matrices/fanfic_to_background.harry_potter.mikolov.v2.nptxt'
    #canon2bg_path = '/usr0/home/qinlans/ACL_2019/transformation_matrices/canon_to_background.harry_potter.mikolov.nptxt'

    #fics_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/fics_paras'
    if input_type == 'assertion':
        input_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/emnlp_dataset_6k/output/assertion_extraction'
    elif input_type == 'quote':
        input_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/emnlp_dataset_6k/output/quote_attribution'

    char_contexts_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/temp/char_contexts_{}_{}.pkl'
    char_embs_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/embeddings/char_vecs_{}_{}_{}.pkl'
    vectorizer_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/temp/{}_vectorizer_{}.pkl'

    # Load selected files for TSNE
    if tsne_fic_ids_path is not None:
        selected_fic_ids = pd.read_csv(tsne_fic_ids_path)['fic_id'].tolist()

    # Load embeddings
    if not vector_combination.startswith('ngrams'):
        print("Loading embeddings...")
        fanfic_embs = KeyedVectors.load_word2vec_format(fanfic_embs_fpath)
        #fanfic_mt_embs = KeyedVectors.load_word2vec_format(fanfic_mt_embs_fpath)
        #canon_mt_embs = KeyedVectors.load_word2vec_format(canon_mt_embs_fpath)

        # ## Align embeddings to background space
        # Load transformation matrices
        if mt_align:
            fanfic2bg = np.loadtxt(fanfic2bg_path)
            #canon2bg = np.loadtxt(canon2bg_path)

        char_vecs = {}

        print("Calculating character embeddings (noncontextual)...")
        for c in chars:
            char_vecs[c] = {}
            if mt_align:
                char_vecs[c]['fanfic'] = np.matmul(fanfic2bg, fanfic_embs[c]) 
                #char_vecs[c]['canon'] = np.matmul(canon2bg, canon_mt_embs[c]) 
                #char_vecs[c]['dist_f-c'] = cosine(char_vecs[c]['fanfic'], char_vecs[c]['canon'])

            else:
                char_vecs[c]['fanfic'] = fanfic_embs[c]

    if extract_contexts: 
        print("Extracting local context for characters...")
        manager = Manager()
        global char_contexts_fanfic
        char_contexts_fanfic = manager.dict()
        for pairing in pairings:
            char_contexts_fanfic[pairing] = manager.dict()

            if context == 'local': # specific to characters, to context windows
                for c in pairing:
                    #char_contexts_fanfic[pairing][c] = {w: {} for w in context_windows}
                    char_contexts_fanfic[pairing][c] = manager.dict()
                    for cw in context_windows:
                        char_contexts_fanfic[pairing][c][cw] = manager.dict()
        
        with Pool(15) as p:
            fnames = sorted(os.listdir(input_fpath))
            params = list(zip(fnames, 
                            itertools.repeat(pairings),
                            itertools.repeat(input_fpath),
                            itertools.repeat(context_windows),
                            itertools.repeat(fic_pairing_paragraph_threshold),
                            itertools.repeat(chars),
                            itertools.repeat(context),
                    ))
            if input_type == 'paragraph':
                #list(tqdm(p.imap(extract_character_local_context, params), total=len(params))) 
                list(tqdm(p.imap(extract_character_paragraph, params), total=len(params))) 
                # TODO: paragraphs with and without local contexts

            elif input_type == 'assertion':
                list(tqdm(p.imap(extract_character_assertion, params), total=len(params))) 
                #list(map(extract_character_assertion, params))

            elif input_type == 'quote':
                list(tqdm(p.imap(extract_character_quote, params), total=len(params))) 
                #list(map(extract_character_quote, params))

        # Save extracted contexts
        print("Saving extracted contexts...")
        char_contexts_fanfic_d = {}
        if context == 'local':
            for pairing in pairings:
                char_contexts_fanfic_d[pairing] = {}
                for c in pairing:
                    char_contexts_fanfic_d[pairing][c] = {}
                    for cw in context_windows:
                        char_contexts_fanfic_d[pairing][c][cw] = char_contexts_fanfic[pairing][c][cw].copy()

        elif context == 'all':
            for pairing in pairings:
                char_contexts_fanfic_d[pairing] = char_contexts_fanfic[pairing].copy()

        with open(char_contexts_outpath.format(input_type, context), 'wb') as f:
            pickle.dump(char_contexts_fanfic_d, f)

    else:
        # Load extracted contexts
        print("Loading extracted contexts...")
        with open(char_contexts_outpath.format(input_type, context), 'rb') as f:
            char_contexts_fanfic = pickle.load(f)
        
    # Ngrams
    if vector_combination == 'ngrams1':
        print("Building fanfic contextualized vectors...")
        char_vecs_per_fic = {}
        for pairing in pairings:
            char_vecs_per_fic[pairing] = {}

            if context == 'local':
                for c in pairing:
                    char_vecs_per_fic[pairing][c] = {w: {} for w in context_windows}

        # Use sklearn's text feature extraction
        corpus = []
        for pairing in tqdm(pairings):

            if context == 'local':
                for c in pairing:
                    for cw in context_windows:
                        for fname, context_wds in sorted(char_contexts_fanfic[pairing][c][cw].items()): # for every fic, sorted
                            if len(context_wds) == 0:
                                continue
                            else:
                                corpus.append(' '.join(context_wds))

            #elif context == 'paragraph' or context == 'assertion':
            elif context == 'all':
                for fname, context_wds in sorted(char_contexts_fanfic[pairing].items()): # for every fic, sorted
                    if len(context_wds) == 0:
                        continue
                    else:
                        corpus.append(' '.join(context_wds))

        print("\tFitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        char_vecs = vectorizer.fit_transform(corpus)

        # Save vectors in dictionary stx
        i = 0
        for pairing in tqdm(pairings):

            if context == 'local':
                for c in pairing:
                    for cw in context_windows:
                        for fname, context_wds in sorted(char_contexts_fanfic[pairing][c][cw].items()): # for every fic, sorted
                            if len(context_wds) == 0:
                                continue
                            else:
                                char_vecs_per_fic[pairing][c][cw][fname] = char_vecs[i]
                                i += 1

            elif context == 'all':
                for fname, context_wds in sorted(char_contexts_fanfic[pairing].items()): # for every fic, sorted
                    if len(context_wds) == 0:
                        continue
                    else:
                        char_vecs_per_fic[pairing][fname] = char_vecs[i]
                        i += 1

        # Save vectorizer
        with open(vectorizer_outpath.format(vector_combination, context), 'wb') as f:
            pickle.dump(vectorizer, f) 

    else: # Not ngrams

        # Get TF-IDF weighting for terms
        print("Getting TF-IDF weights...")
        idf_weights_fanfic = {}

        for pairing in tqdm(pairings):
            idf_weights_fanfic[pairing] = {}

            if context == 'local':
                for c in pairing:
                    idf_weights_fanfic[pairing][c] = {}

                    for cw in context_windows:
                        context_toks = [' '.join(toks) for toks in char_contexts_fanfic[pairing][c][cw].values()]
                        if len(context_toks) == 0:
                            continue
                    
                        vectorizer = TfidfVectorizer(stop_words=stopwords)
                        try:
                            vectorizer.fit(context_toks)
                            idf_weights_fanfic[pairing][c][cw] = (vectorizer.vocabulary_, vectorizer.idf_)
                        except ValueError as e:
                            del idf_weights_fanfic[pairing][c]

        # Get contextualized fanfic vectors
        print("Building fanfic contextualized vectors...")

        char_vecs_per_fic = {}
        for pairing in pairings:
            char_vecs_per_fic[pairing] = {}

            if context == 'local':
                for c in pairing:
                    char_vecs_per_fic[pairing][c] = {w: {} for w in context_windows}

        for pairing in tqdm(pairings):

            if context == 'local':
                for c in pairing:
                    for cw in context_windows:
                        for fname, context_wds in char_contexts_fanfic[pairing][c][cw].items(): # for every fic, sorted
                            if len(context_wds) == 0 or not c in idf_weights_fanfic[pairing]:
                                continue
                            wd_indices, wd_weights = idf_weights_fanfic[pairing][c][cw]

                            context_embs = [fanfic_embs[w] * wd_weights[wd_indices[w]] for w in context_wds if w in fanfic_embs and w in wd_indices]

                            if len(context_embs) == 0: continue
                            context_vec = np.mean(context_embs, axis=0)
                            
                            if vector_combination == 'context_only':
                                char_vec = context_vec
                            
                            elif vector_combination == 'add':
                                char_vec = np.add(char_vecs[c]['fanfic'], context_vec)

                            if (not vector_combination.startswith('ngrams')) and (char_vec.shape != (100,)): continue # only context words found are not in vocabulary

                            char_vecs_per_fic[pairing][c][cw][fname] = char_vec

            elif context == 'all':
                for fname, context_wds in char_contexts_fanfic[pairing].items(): # for every fic, sorted
                    if len(context_wds) == 0 or not pairing in idf_weights_fanfic:
                        continue
                    wd_indices, wd_weights = idf_weights_fanfic[pairing]

                    #context_embs = [fanfic_embs[w] * wd_weights[wd_indices[w]] for w in context_wds if w in fanfic_embs and w in wd_indices]

                    if len(context_embs) == 0: continue
                    context_vec = np.mean(context_embs, axis=0)
                    
                    if vector_combination == 'context_only':
                        char_vec = context_vec
                    
                    elif vector_combination == 'add':
                        char_vec = np.add(char_vecs[c]['fanfic'], context_vec)

                    if not vector_combination.startswith('ngrams') and char_vec.shape != (100,): continue

                    char_vecs_per_fic[pairing][fname] = char_vec


    # Save embeddings
    with open(char_embs_outpath.format(input_type, context, vector_combination), 'wb') as f:
        pickle.dump(char_vecs_per_fic, f)

if __name__ == '__main__': main()
