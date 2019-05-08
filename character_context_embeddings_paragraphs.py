from gensim.models import KeyedVectors, FastText 
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm as tqdm
import pickle
import pdb
import argparse
import re
from multiprocessing import Pool, Manager, Process
import itertools
from nltk.tokenize import sent_tokenize


def pairing_present(text, pairing):
    if re.search(r'\b{}\b'.format(pairing[0]), text) and re.search(r'\b{}\b'.format(pairing[1]), text):
        return True
    else:
        return False


def extract_character_local_context(params):
    fname = params[0]
    pairings = params[1]
    fics_fpath = params[2]
    context_windows = params[3]
    fic_pairing_paragraph_threshold = params[4]
    save_paras_fpath = params[5]

    fic_id = int(fname.split('_')[0])
    
    with open(os.path.join(fics_fpath, fname)) as f:
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

    fic_pairing_paragraph_threshold = 5
    context_windows = [10] # before and after, so total window is this value * 2
    mt_align = False
    extract_contexts = False
    #save_paras_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/pairings/{}/{}_paras.txt' # None for not saving this
    save_paras_fpath = None

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
    #fanfic_mt_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/fanfic.harry_potter.1k-5k.v2.lower.model.vec'
    #canon_mt_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/canon.harry_potter.lower.model.vec'
    #fanfic2bg_path = '/usr0/home/qinlans/ACL_2019/transformation_matrices/fanfic_to_background.harry_potter.mikolov.v2.nptxt'
    #canon2bg_path = '/usr0/home/qinlans/ACL_2019/transformation_matrices/canon_to_background.harry_potter.mikolov.nptxt'

    fics_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/fics_paras'
    #canon_fpath = '/usr0/home/jfiacco/Research/fanfic/canon_data/harry_potter_tokenized/'

    char_contexts_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/temp/char_contexts.pkl'
    char_embs_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/embeddings/char_vecs_lc_{}_paragraphs.pkl'

    # Load selected files for TSNE
    if tsne_fic_ids_path is not None:
        selected_fic_ids = pd.read_csv(tsne_fic_ids_path)['fic_id'].tolist()

    # Load embeddings
    print("Loading embeddings...")
    fanfic_embs = KeyedVectors.load_word2vec_format(fanfic_embs_fpath)
    #fanfic_mt_embs = KeyedVectors.load_word2vec_format(fanfic_mt_embs_fpath)
    #canon_mt_embs = KeyedVectors.load_word2vec_format(canon_mt_embs_fpath)

    # ## Align embeddings to background space
    # Load transformation matrices
    if mt_align:
        fanfic2bg = np.loadtxt(fanfic2bg_path)
        canon2bg = np.loadtxt(canon2bg_path)

    char_vecs = {}

    print("Calculating character embeddings (noncontextual)...")
    for c in chars:
        char_vecs[c] = {}
        if mt_align:
            char_vecs[c]['fanfic'] = np.matmul(fanfic2bg, fanfic_mt_embs[c]) 
            char_vecs[c]['canon'] = np.matmul(canon2bg, canon_mt_embs[c]) 
            char_vecs[c]['dist_f-c'] = cosine(char_vecs[c]['fanfic'], char_vecs[c]['canon'])

        else:
            char_vecs[c]['fanfic'] = fanfic_embs[c]

    # ## Local context for fanfic

    # # Calculate local context character name vectors
    #char_contexts_fanfic = {}
    #for pairing in pairings:
    #    char_contexts_fanfic[pairing] = {}
    #    for c in chars:
    #        char_contexts_fanfic[pairing][c] = {w: {} for w in context_windows}
    
    if extract_contexts: 
        print("Extracting local context for characters...")
        manager = Manager()
        global char_contexts_fanfic
        char_contexts_fanfic = manager.dict()
        for pairing in pairings:
            char_contexts_fanfic[pairing] = manager.dict()
            for c in pairing:
                #char_contexts_fanfic[pairing][c] = {w: {} for w in context_windows}
                char_contexts_fanfic[pairing][c] = manager.dict()
                for cw in context_windows:
                  char_contexts_fanfic[pairing][c][cw] = manager.dict()
        
        with Pool(15) as p:
            fnames = sorted(os.listdir(fics_fpath))
            params = list(zip(fnames, 
                            itertools.repeat(pairings),
                            itertools.repeat(fics_fpath),
                            itertools.repeat(context_windows),
                            itertools.repeat(fic_pairing_paragraph_threshold),
                            itertools.repeat(save_paras_fpath),
                    ))
            list(tqdm(p.imap(extract_character_local_context, params), total=len(params))) 

        # Save extracted contexts
        print("Saving extracted contexts...")
        char_contexts_fanfic_d = {}
        for pairing in pairings:
            char_contexts_fanfic_d[pairing] = {}
            for c in pairing:
                char_contexts_fanfic_d[pairing][c] = {}
                for cw in context_windows:
                    char_contexts_fanfic_d[pairing][c][cw] = char_contexts_fanfic[pairing][c][cw].copy()

        with open(char_contexts_outpath, 'wb') as f:
            pickle.dump(char_contexts_fanfic_d, f)

    else:
        # Load extracted contexts
        print("Loading extracted contexts...")
        with open(char_contexts_outpath, 'rb') as f:
            char_contexts_fanfic = pickle.load(f)
        

#    for fname in tqdm(sorted(os.listdir(fics_fpath))):
#    #for fname in tqdm(sorted(os.listdir(fics_fpath))[:1000]):
#        fic_id = int(fname.split('_')[0])
#        
#        with open(os.path.join(fics_fpath, fname)) as f:
#            paras = [p.split() for p in f.read().splitlines()]
#
#            #for c in chars:
#            for pairing in pairings:
#
#                fic_pairing_count = 0
#                fic_contexts = {w: [] for w in context_windows}
#                    
#                for para in paras:
#
#                    # Determine whether both characters are in the paragraph
#                    if pairing_present(' '.join(para), pairing):
#
#                        fic_pairing_count += 1
#
#                        for c in pairing:
#                            for idx in [i for i,token in enumerate(para) if token==c]:
#                                for context_window in context_windows:
#                                    
#                                    fic_contexts[context_window] += para[max(0, idx-context_window) : idx] # before
#                                    fic_contexts[context_window] += para[idx+1 : min(idx+1+context_window, len(para))] # after
#            
#                if fic_pairing_count >= fic_pairing_paragraph_threshold:
#                    for c in pairing:
#                        for context_window in context_windows:
#                            char_contexts_fanfic[pairing][c][context_window][fic_id] = fic_contexts[context_window]


    # Build, store aligned fanfic vectors
    #fanfic_aligned = {}
    #print("Building aligned embeddings...")

    #for pairing in tqdm(pairings):
    #    for c in pairing:
    #        for context_window in context_windows:
    #            for fname, context_wds in char_contexts_fanfic[pairing][c][context_window].items():
    #                context_wds = set(context_wds)
    #                for w in context_wds:
    #                    if not w in fanfic_aligned and w in fanfic_embs:
    #                        fanfic_aligned[w] = np.matmul(fanfic2bg, fanfic_mt_embs[w])


    # Ngrams
    if vector_combination == 'ngrams1':
        print("Building fanfic contextualized vectors...")
        char_vecs_per_fic = {}
        for pairing in pairings:
            char_vecs_per_fic[pairing] = {}
            for c in pairing:
                char_vecs_per_fic[pairing][c] = {w: {} for w in context_windows}

        # Use sklearn's text feature extraction
        corpus = []
        for pairing in tqdm(pairings):
            for c in pairing:
                for cw in context_windows:
                    for fname, context_wds in sorted(char_contexts_fanfic[pairing][c][cw].items()): # for every fic, sorted
                        if len(context_wds) == 0:
                            continue
                        else:
                            corpus.append(' '.join(context_wds))

        print("\tFitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(stop_words='english')
        char_vecs = vectorizer.fit_transform(corpus)

        # Save vectors in dictionary stx
        i = 0
        for pairing in tqdm(pairings):
            for c in pairing:
                for cw in context_windows:
                    for fname, context_wds in sorted(char_contexts_fanfic[pairing][c][cw].items()): # for every fic, sorted
                        if len(context_wds) == 0:
                            continue
                        else:
                            char_vecs_per_fic[pairing][c][cw][fname] = char_vecs[i]
                            i += 1

    else: # Not ngrams

        # Get TF-IDF weighting for terms
        print("Getting TF-IDF weights...")
        idf_weights_fanfic = {}

        for pairing in tqdm(pairings):
            idf_weights_fanfic[pairing] = {}
            for c in pairing:
                idf_weights_fanfic[pairing][c] = {}

                for cw in context_windows:
                    context_toks = [' '.join(toks) for toks in char_contexts_fanfic[pairing][c][cw].values()]
                    if len(context_toks) == 0:
                        continue
                
                    vectorizer = TfidfVectorizer(stop_words='english')
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
            for c in pairing:
                char_vecs_per_fic[pairing][c] = {w: {} for w in context_windows}

        for pairing in tqdm(pairings):
            for c in pairing:
                for cw in context_windows:
                    for fname, context_wds in char_contexts_fanfic[pairing][c][cw].items(): # for every fic, sorted
                        if len(context_wds) == 0 or not c in idf_weights_fanfic[pairing]:
                            continue
                        wd_indices, wd_weights = idf_weights_fanfic[pairing][c][cw]

                        if vector_combination == 'ngrams1':

                            # Redone with sklearn text feature extraction
                            vocab = set(wd_indices.keys()).intersection(set(fanfic_embs.wv.vocab.keys()))
                            word2index = {w: i for w,i in zip(sorted(vocab), range(len(vocab)))}                        

                            bow = np.zeros(len(vocab))
                            for w in context_wds:
                                if w in fanfic_embs and w in wd_indices:
                                    bow[word2index[w]] += wd_weights[wd_indices[w]]

                        else:
                            context_embs = [fanfic_embs[w] * wd_weights[wd_indices[w]] for w in context_wds if w in fanfic_embs and w in wd_indices]

                            if len(context_embs) == 0: continue
                            context_vec = np.mean(context_embs, axis=0)
                        
                        if vector_combination == 'context_only':
                            char_vec = context_vec
                        
                        elif vector_combination == 'add':
                            char_vec = np.add(char_vecs[c]['fanfic'], context_vec)

                        elif vector_combination == 'ngrams1':
                            char_vec = bow

                        if (not vector_combination.startswith('ngrams')) and (char_vec.shape != (100,)): continue # only context words found are not in vocabulary

                        char_vecs_per_fic[pairing][c][cw][fname] = char_vec


    # ## Local context for canon
#    print("Running everything for canon...")
#    char_contexts_canon = {}
#    for c in chars:
#        char_contexts_canon[c] = {w: {} for w in context_windows}
#
#    for fname in tqdm(sorted(os.listdir(canon_fpath))):
#        
#        with open(os.path.join(canon_fpath, fname)) as f:
#            paras = [p.lower().split() for p in f.read().splitlines()]
#            for c in chars:
#    #         for c in minor_chars:
#                
#                fic_contexts = {w: [] for w in context_windows}
#                
#                for para in paras:
#                    for idx in [i for i,token in enumerate(para) if token==c]:
#                        for context_window in context_windows:
#                            
#                            fic_contexts[context_window] += para[max(0, idx-context_window) : idx] # before
#                            fic_contexts[context_window] += para[idx+1 : min(idx+1+context_window, len(para))] # after
#        
#                for context_window in context_windows:
#                    char_contexts_canon[c][context_window][fname] = fic_contexts[context_window]
#
#    # Build, store aligned canon vectors
#    canon_aligned = {}
#
#    for c in tqdm(chars):
#        for context_window in context_windows:
#            for fname, context_wds in char_contexts_canon[c][context_window].items():
#                context_wds = set(context_wds)
#                for w in context_wds:
#                    if not w in canon_aligned and w in canon_mt_embs:
#                        canon_aligned[w] = np.matmul(canon2bg, canon_mt_embs[w])
#
#    # Get TF-IDF weighting for terms
#    idf_weights_canon = {}
#
#    for c in tqdm(chars):
#        idf_weights_canon[c] = {}
#        
#        for cw in context_windows:
#            vectorizer = TfidfVectorizer(stop_words='english')
#            vectorizer.fit([' '.join(toks) for toks in char_contexts_canon[c][cw].values()])
#            idf_weights_canon[c][cw] = (vectorizer.vocabulary_, vectorizer.idf_)
#
#    # Get contextualized canon vectors
#    char_vecs_per_canon = {}
#    for c in chars:
#        char_vecs_per_canon[c] = {w: {} for w in context_windows}
#
#    for c in tqdm(chars):
#        for cw in context_windows:
#            for fname, context_wds in char_contexts_canon[c][cw].items(): # for every canon, sorted
#                if len(context_wds) == 0: continue
#                    
#                wd_indices, wd_weights = idf_weights_canon[c][cw]
#                context_embs = [canon_aligned[w] * wd_weights[wd_indices[w]] for w in context_wds if                             w in canon_mt_embs and w in wd_indices]
#                context_vec = np.mean(context_embs, axis=0)
#                
#    #             char_vec = np.hstack([char_vecs[c]['canon'], context_vec])
#    #             if char_vec.shape != (200,): continue # only context words found are not in vocabulary
#    #             char_vec = np.mean([char_vecs[c]['canon'], context_vec], axis=0)
#
#                #char_vec = context_vec
#                char_vec = np.add(char_vecs[c]['canon'], context_vec)
#                if char_vec.shape != (100,): continue # only context words found are not in vocabulary
#                char_vecs_per_canon[c][cw][fname] = char_vec
#
#            char_vecs[c][f'canon_lc{cw}_context'] = np.mean(list(char_vecs_per_canon[c][cw].values()), axis=0)
#            char_vecs[c][f'canon_lc{cw}_per_story_{vector_combination}'] = char_vecs_per_canon[c][cw]
#
#
#    # ## Distances canon-fanfic
#    for c in chars:
#        for cw in context_windows:
#            char_vecs[c][f'dist_lc{cw}_context_f-c'] = cosine(char_vecs[c][f'fanfic_lc{cw}_context'], char_vecs[c][f'canon_lc{cw}_context'])

    # Save distances, embeddings
    for cw in context_windows:
        with open(char_embs_outpath.format(vector_combination), 'wb') as f:
            pickle.dump(char_vecs_per_fic, f)


    # ## Reduce dimensions
    if run_tsne:
        
        all_char_vecs = []
        char_labels = []
        fname_labels = []

        # Fanfic
        for c in chars:

            for fname, vec in sorted(char_vecs_per_fic[c][10].items()) + sorted(char_vecs_per_canon[c][10].items()):
                if tsne_fic_ids_path is None or fname in selected_fic_ids:
                    all_char_vecs.append(vec)
                    char_labels.append(c)
                    fname_labels.append(fname)

        print("Running PCA...")
        pca = PCA(n_components=20)
        reduced = pca.fit_transform(all_char_vecs)

        print("Running TSNE...")
        tsne = TSNE(n_components=2)
        tsne_reduced = tsne.fit_transform(reduced)

        np.savetxt(tsne_char_embs_outpath, tsne_reduced)
        with open(tsne_char_labels_outpath, 'w') as f:
            for char in char_labels:
                f.write(f"{char}\n")
        with open(tsne_fname_labels_outpath, 'w') as f:
            for fname in fname_labels:
                f.write(f"{fname}\n")

    #print("Plotting characters...")
    #chars2i = {c: i for i,c in enumerate(chars)}
    #cdict = {'draco': 'red', 'hermione': 'green', 'harry': 'blue', 'ginny': 'yellow', 'ron': 'purple'}
    ##selected_chars = ['ginny', 'harry']

    #fig, ax = plt.subplots(figsize=(10,8))
    #for c in cdict:
    #    ix = np.where(np.array(char_labels) == c)
    #    ax.scatter(tsne_reduced[:,0][ix], tsne_reduced[:,1][ix], c=cdict[c], s=1, label=c)
    #    
    #ax.legend()
    #plt.show()

    #plt.savefig(plot_outpath)

if __name__ == '__main__': main()
