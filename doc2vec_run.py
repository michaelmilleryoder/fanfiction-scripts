import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from random import shuffle
from collections import OrderedDict
import multiprocessing
import os, sys
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

    def load_data(self, data_fpath, data_name, multi=False):
        self.data_name = data_name
        if multi:
            self.data_fpath = [os.path.join(self.base_dirpath, data_fpath.format(name)) for name in data_name]
        else:
            self.data_fpath = os.path.join(self.base_dirpath, data_fpath.format(data_name))
        if multi:
            self.data = pd.concat([pd.read_pickle(f) for f in self.data_fpath])
        else:
            self.data = pd.read_pickle(self.data_fpath)

    def build_tagged_docs(self, text_colname, labels_colname):
        #self.data.fillna(value={text_colname: '', labels_colname: [None]}, inplace=True)
        #self.data.fillna(value={text_colname: ''}, inplace=True)

        # Get rid of NAN rows
        self.data.dropna(subset=[text_colname, labels_colname], how='all', inplace=True)
        print(f"{len(self.data)} rows of data considered.")

        alldocs = []    

        for line, tags in zip(self.data[text_colname], self.data[labels_colname]):
            if isinstance(tags, float):
                pdb.set_trace()
            tokens = gensim.utils.to_unicode(str(line)).split()
            alldocs.append(TaggedDocument(tokens, tags))

        doc_list = alldocs[:]  
        shuffle(doc_list)

        self.tagged_doc = doc_list


class ModelTrainer():

    def __init__(self, data_loader, cores=20):
        self.cores = cores
        self.models = []
        self.data_loader = data_loader
        self.data_name = self.data_loader.base_dirpath
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

    def train_models(self, model_name):
        for model in self.models: 
            print("Training %s" % model)
            model.train(self.data_loader.tagged_doc, total_examples=len(self.data_loader.tagged_doc), epochs=model.epochs) # Adjust epochs in model spec

            print("Saving %s" % model)
            model_outdir = os.path.join(self.model_dirpath, model_name)
            if not os.path.exists(model_outdir):
                os.makedirs(model_outdir)
            model_outpath = os.path.join(model_outdir, f"{model.comment}.model")
            model.save(model_outpath)
            print("Saved to %s" % model_outpath)

def main():
    base_dirpath = '/usr2/mamille2/fanfiction-project'
    #data_name = 'detroit'
    data_names = ['academia', 'detroit', 'friends']
    data_format_fpath = 'data/ao3/{0}/{0}_discoursedb_data.pkl'
    multi = True

    print("Loading data...", end='')
    sys.stdout.flush()
    data_loader = DataLoader(base_dirpath)
    data_loader.load_data(data_format_fpath, data_names, multi=multi)
    #data_loader.build_tagged_docs('preprocessed_content', 'relationship_type')
    data_loader.build_tagged_docs('preprocessed_text', 'category')

    model_trainer = ModelTrainer(data_loader)
    model_trainer.build_models()
    model_name = 'academia-detroit-friends'
    #model_name = data_loader.data_name
    model_trainer.train_models(model_name)

    # models_by_name = OrderedDict((str(model), model) for model in simple_models)
    # models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
    # models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])

if __name__ == '__main__':
    main()
