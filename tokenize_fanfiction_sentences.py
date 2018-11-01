import spacy
from tqdm import tqdm

nlp = spacy.load('en')

# Load fanfiction data
fandoms = [
    'detroit',
    'friends',
    'academia',
    'allmarvel',
]

data_fpath = '/usr0/home/mamille2/erebor/fanfiction-project/data/ao3/{0}/ao3_{0}_sentences.txt'

for f in fandoms:
    print(f)

    with open(data_fpath.format(f)) as file_obj:
        sentences = file_obj.read().splitlines()
        sent_toks = []
        for sent in tqdm(sentences):
            sent_toks.append([tok.text for tok in nlp.tokenizer(sent.lower())])

    with open(data_fpath.format(f) + '.tokens', 'w') as wfile:
        for sent_tok in sent_toks:
            wfile.write(' '.join(sent_tok) + '\n')
