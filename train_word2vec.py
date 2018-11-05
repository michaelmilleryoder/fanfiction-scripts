from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import os
data_dirpath = '/usr0/home/mamille2/erebor'


def main():
    # # Train word2vec on fanfiction data, initialized with Google News embeddings

    # ### Add in fanfiction data

    fandoms = [
        'allmarvel',
    ]

    # Load pretrained background embeddings
    print("Loading pretrained embeddings...")
    pretrained_fpath = os.path.join(data_dirpath, 'word_embeddings', 'GoogleNews-vectors-negative300.bin')
    pretrained_wv = KeyedVectors.load_word2vec_format(pretrained_fpath, binary=True)

    for f in fandoms:
        print(f)
        sentences_fpath = os.path.join(data_dirpath, 'fanfiction-project/data/ao3', f, f'ao3_{f}_sentences.txt')

        # Load fanfiction sentences (in RAM)
        #with open(sentences_fpath) as file_obj:
        #    sentences = file_obj.read().splitlines()
        #    print(len(sentences))
        #sentences = [s.split() for s in sentences]

        # Load fanfiction sentences (in RAM)
        sentences = LineSentence(sentences_fpath)
        sentences_len = 126219476 # from bash wc -l

        # ## Create model initialized with Google News 300-d embeddings

        print("Building vocab...")
        model = Word2Vec(size=300, min_count=5)
        model.build_vocab(sentences)

        model.build_vocab([list(pretrained_wv.vocab.keys())], update=True) # should add words, though doesn't seem to
        model.intersect_word2vec_format(pretrained_fpath, lockf=1.0, binary=True)

        # Train model
        print('Training model...')
        model.train(sentences, total_examples=sentences_len, epochs=5)

        # Save model
        print('Saving model...')
        model.save(os.path.join(data_dirpath, 'word_embeddings', f'{f}_GoogleNews_300d.model'))

        # ## Save embeddings in txt format
        print('Saving embeddings...')
        emb_outpath = os.path.join(data_dirpath, 'word_embeddings', f'{f}_GoogleNews_300d.txt')
        model.wv.save_word2vec_format(emb_outpath), binary=False)
        print(f'Saved embeddings to {emb_outpath}')
        print()

if __name__ == '__main__':
    main()
