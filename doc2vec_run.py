import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from random import shuffle
from collections import OrderedDict
import multiprocessing
import os
import pdb
import pandas as pd

""" Train doc2vec to build vectors for various metadata labels. Save models """

class DataLoader():

    def __init__(self, base_dirpath):
        # I/O
        self.base_dirpath = base_dirpath
        self.test_fraction = 0.1
        self.tagged_doc = None
        self.data = None

    def load_data(self, data_fpath):
        self.data_fpath = os.path.join(self.base_dirpath, data_fpath)
        self.data = pd.read_pickle(self.data_fpath)

    def build_tagged_docs(self, text_colname, labels_colname):
        alldocs = []
        for line, tags in zip(self.data[text_colname], self.data[labels_colname]):
            tokens = gensim.utils.to_unicode(line).split()
            alldocs.append(TaggedDocument(tokens, tags))

        doc_list = alldocs[:]  
        shuffle(doc_list)

        self.tagged_doc = doc_list


class ModelTrainer():

    def __init__(self, data_loader, cores=20):
        self.cores = cores
        self.models = []
        self.data_loader = data_loader
        self.model_dirpath = os.path.join(data_loader.base_dirpath, 'models')
        pass

    def build_models(self):
        assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
        self.models = [
            # PV-DBOW plain
            Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, 
                    epochs=20, workers=self.cores, comment='PV-DBOW_d100n5mc2t20'),
            # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
            Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, 
                    epochs=20, workers=self.cores, alpha=0.05, comment='PV-DM_d100w10n5mc2t20alpha0.05'),
            # PV-DM w/ concatenation - big, slow, experimental mode
            # window=5 (both sides) approximates paper's apparent 10-word total window size
            Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, sample=0, 
                    epochs=20, workers=self.cores, comment='PV-DM_concatd100w5n5mc2t20'),
        ]

        for model in self.models:
            model.build_vocab(self.data_loader.tagged_doc)
            print("%s vocabulary scanned & state initialized" % model)

    def train_models(self):
        for model in self.models: 
            print("Training %s" % model)
            model.train(self.data_loader.tagged_doc, total_examples=len(self.data_loader.tagged_doc), epochs=model.epochs) # Adjust epochs in model spec

            print("Saving %s" % model)
            model.save(os.path.join(self.model_dirpath, f"{model.comment}.model"))

def main():
    base_dirpath = '/usr2/mamille2/fanfiction-project'
    data_fpath = 'data/ao3/friends/friends_discoursedb_data.pkl'
    data_loader = DataLoader(base_dirpath)
    data_loader.load_data(data_fpath)
    data_loader.build_tagged_docs('preprocessed_content', 'relationship_type')

    model_trainer = ModelTrainer(data_loader)
    model_trainer.build_models()
    model_trainer.train_models()

    # models_by_name = OrderedDict((str(model), model) for model in simple_models)
    # models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
    # models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])

if __name__ == '__main__':
    main()
