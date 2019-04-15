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


def main():

    # I/O
    fanfic_mt_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/fanfic.harry_potter.1k-5k.v2.lower.model.vec'
    canon_mt_embs_fpath = '/usr0/home/jfiacco/Research/fanfic/embeddings/canon.harry_potter.lower.model.vec'
    fanfic2bg_path = '/usr0/home/qinlans/ACL_2019/transformation_matrices/fanfic_to_background.harry_potter.mikolov.v2.txt'
    canon2bg_path = '/usr0/home/qinlans/ACL_2019/transformation_matrices/canon_to_background.harry_potter.mikolov.txt'

    fics_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/harrypotter/dataset_1k-5k/filtered_paras'
    canon_fpath = '/usr0/home/jfiacco/Research/fanfic/canon_data/harry_potter_tokenized/'

    # TSNE settings
    #tsne_fic_ids_path = None # if None, then run TSNE on all fics
    #tsne_fic_ids_path = '/usr0/home/mamille2/erebor/fanfiction-project/data/qian_sample1_fic_ids.csv' # if None, then run TSNE on all fics
    tsne_char_embs_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_chars_lc10_context.nptxt'
    tsne_char_labels_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_chars_lc10_context_labels.txt'
    tsne_fname_labels_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_fnames_lc10_context.txt'
    #plot_outpath = '/usr0/home/mamille2/erebor/fanfiction-project/output/tsne_chars_lc10_context_plot.png'

    # Settings
    chars = [
        'harry',
        'hermione',
        'ron',
        'ginny',
        'draco',
        'neville',
        'luna',
        'remus',
        'sirius',
        'severus',
        'james',
        'lily',
    ]

    # Load selected files for TSNE
    if tsne_fic_ids_path is not None:
        selected_fic_ids = pd.read_csv(tsne_fic_ids_path)['fic_id'].tolist()

    # Load embeddings
    print("Loading embeddings...")
    fanfic_mt_embs = KeyedVectors.load_word2vec_format(fanfic_mt_embs_fpath)
    canon_mt_embs = KeyedVectors.load_word2vec_format(canon_mt_embs_fpath)

    # ## Align embeddings to background space
    # Load transformation matrices
    fanfic2bg = np.loadtxt(fanfic2bg_path)
    canon2bg = np.loadtxt(canon2bg_path)

    char_vecs = {}

    print("Calculating character embeddings (noncontextual)...")
    for c in chars:
        char_vecs[c] = {}
        char_vecs[c]['fanfic'] = np.matmul(fanfic2bg, fanfic_mt_embs[c]) 
        char_vecs[c]['canon'] = np.matmul(canon2bg, canon_mt_embs[c]) 
        char_vecs[c]['dist_f-c'] = cosine(char_vecs[c]['fanfic'], char_vecs[c]['canon'])


    # # Calculate local context character name vectors

    #context_windows = [10, 25, 50] # before and after, so total window is this value * 2
    context_windows = [10] # before and after, so total window is this value * 2

    # ## Local context for fanfic
    print("Extracting local context for characters...")
    char_contexts_fanfic = {}
    for c in chars:
        char_contexts_fanfic[c] = {w: {} for w in context_windows}

    for fname in tqdm(sorted(os.listdir(fics_fpath))):
        
        with open(os.path.join(fics_fpath, fname)) as f:
            paras = [p.split() for p in f.read().splitlines()]
            for c in chars:
    #         for c in minor_chars:
                
                fic_contexts = {w: [] for w in context_windows}
                
                for para in paras:
                    for idx in [i for i,token in enumerate(para) if token==c]:
                        for context_window in context_windows:
                            
                            fic_contexts[context_window] += para[max(0, idx-context_window) : idx] # before
                            fic_contexts[context_window] += para[idx+1 : min(idx+1+context_window, len(para))] # after
        
                for context_window in context_windows:
                    char_contexts_fanfic[c][context_window][fname] = fic_contexts[context_window]


    # Build, store aligned fanfic vectors
    fanfic_aligned = {}
    print("Building aligned embeddings...")

    for c in tqdm(chars):
        for context_window in context_windows:
            for fname, context_wds in char_contexts_fanfic[c][context_window].items():
                context_wds = set(context_wds)
                for w in context_wds:
                    if not w in fanfic_aligned and w in fanfic_mt_embs:
                        fanfic_aligned[w] = np.matmul(fanfic2bg, fanfic_mt_embs[w])


    # Get TF-IDF weighting for terms
    print("Getting TF-IDF weights...")
    idf_weights_fanfic = {}

    for c in tqdm(chars):
        idf_weights_fanfic[c] = {}
        
        for cw in context_windows:
            vectorizer = TfidfVectorizer(stop_words='english')
            vectorizer.fit([' '.join(toks) for toks in char_contexts_fanfic[c][cw].values()])
            idf_weights_fanfic[c][cw] = (vectorizer.vocabulary_, vectorizer.idf_)

    # Get contextualized fanfic vectors
    print("Building fanfic contextualized vectors...")

    char_vecs_per_fic = {}
    for c in chars:
        char_vecs_per_fic[c] = {w: {} for w in context_windows}

    for c in tqdm(chars):
        for cw in context_windows:
            for fname, context_wds in char_contexts_fanfic[c][cw].items(): # for every fic, sorted
                if len(context_wds) == 0: continue
                    
                wd_indices, wd_weights = idf_weights_fanfic[c][cw]
                context_embs = [fanfic_aligned[w] * wd_weights[wd_indices[w]] for w in context_wds if                             w in fanfic_mt_embs and w in wd_indices]
                if len(context_embs) == 0: continue
                context_vec = np.mean(context_embs, axis=0)
                
    #             char_vec = np.hstack([char_vecs[c]['fanfic'], context_vec])
    #             if char_vec.shape != (200,): continue # only context words found are not in vocabulary
    #             char_vec = np.mean([char_vecs[c]['fanfic'], context_vec], axis=0)

                #char_vec = context_vec
                char_vec = np.add(char_vecs[c]['fanfic'], context_vec)

                if char_vec.shape != (100,): continue # only context words found are not in vocabulary
                char_vecs_per_fic[c][cw][fname] = char_vec

            char_vecs[c][f'fanfic_lc{cw}_context'] = np.mean(list(char_vecs_per_fic[c][cw].values()), axis=0)
            char_vecs[c][f'fanfic_lc{cw}_context_per_fic'] = char_vecs_per_fic[c][cw]


    # ## Local context for canon
    print("Running everything for canon...")
    char_contexts_canon = {}
    for c in chars:
        char_contexts_canon[c] = {w: {} for w in context_windows}

    for fname in tqdm(sorted(os.listdir(canon_fpath))):
        
        with open(os.path.join(canon_fpath, fname)) as f:
            paras = [p.lower().split() for p in f.read().splitlines()]
            for c in chars:
    #         for c in minor_chars:
                
                fic_contexts = {w: [] for w in context_windows}
                
                for para in paras:
                    for idx in [i for i,token in enumerate(para) if token==c]:
                        for context_window in context_windows:
                            
                            fic_contexts[context_window] += para[max(0, idx-context_window) : idx] # before
                            fic_contexts[context_window] += para[idx+1 : min(idx+1+context_window, len(para))] # after
        
                for context_window in context_windows:
                    char_contexts_canon[c][context_window][fname] = fic_contexts[context_window]

    # Build, store aligned canon vectors
    canon_aligned = {}

    for c in tqdm(chars):
        for context_window in context_windows:
            for fname, context_wds in char_contexts_canon[c][context_window].items():
                context_wds = set(context_wds)
                for w in context_wds:
                    if not w in canon_aligned and w in canon_mt_embs:
                        canon_aligned[w] = np.matmul(canon2bg, canon_mt_embs[w])

    # Get TF-IDF weighting for terms
    idf_weights_canon = {}

    for c in tqdm(chars):
        idf_weights_canon[c] = {}
        
        for cw in context_windows:
            vectorizer = TfidfVectorizer(stop_words='english')
            vectorizer.fit([' '.join(toks) for toks in char_contexts_canon[c][cw].values()])
            idf_weights_canon[c][cw] = (vectorizer.vocabulary_, vectorizer.idf_)

    # Get contextualized canon vectors
    char_vecs_per_canon = {}
    for c in chars:
        char_vecs_per_canon[c] = {w: {} for w in context_windows}

    for c in tqdm(chars):
        for cw in context_windows:
            for fname, context_wds in char_contexts_canon[c][cw].items(): # for every canon, sorted
                if len(context_wds) == 0: continue
                    
                wd_indices, wd_weights = idf_weights_canon[c][cw]
                context_embs = [canon_aligned[w] * wd_weights[wd_indices[w]] for w in context_wds if                             w in canon_mt_embs and w in wd_indices]
                context_vec = np.mean(context_embs, axis=0)
                
    #             char_vec = np.hstack([char_vecs[c]['canon'], context_vec])
    #             if char_vec.shape != (200,): continue # only context words found are not in vocabulary
    #             char_vec = np.mean([char_vecs[c]['canon'], context_vec], axis=0)

                #char_vec = context_vec
                char_vec = np.add(char_vecs[c]['canon'], context_vec)
                if char_vec.shape != (100,): continue # only context words found are not in vocabulary
                char_vecs_per_canon[c][cw][fname] = char_vec

            char_vecs[c][f'canon_lc{cw}_context'] = np.mean(list(char_vecs_per_canon[c][cw].values()), axis=0)
            char_vecs[c][f'canon_lc{cw}_context_per_story'] = char_vecs_per_canon[c][cw]


    # ## Distances canon-fanfic
    for c in chars:
        for cw in context_windows:
            char_vecs[c][f'dist_lc{cw}_context_f-c'] = cosine(char_vecs[c][f'fanfic_lc{cw}_context'], char_vecs[c][f'canon_lc{cw}_context'])

    # Save distances, embeddings
    with open('/usr0/home/mamille2/erebor/fanfiction-project/embeddings/char_vecs_lc_tfidf.pkl', 'wb') as f:
        pickle.dump(char_vecs, f)


    # ## Reduce dimensions
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
