import os
from tqdm import tqdm as tqdm
import spacy
data_dirpath = '/usr0/home/mamille2/erebor/nyt/'

# ## Compile large sample of words
# Load NYT data
fpath = os.path.join(data_dirpath, 'nyt1994-200206')

print("Loading data")
with open(fpath) as f:
    lines = f.read().splitlines()
    
print(len(lines))

# Strip HTML
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

print("Stripping HTML")
preprocessed = [strip_tags(l.lower()) for l in tqdm(lines)]

with open(os.path.join(data_dirpath, 'nyt_nohtml.txt'), 'w') as f:
    for line in preprocessed:
        f.write(line + '\n')

# Tokenize
print("Tokenizing...")
nlp = spacy.load('en')
preprocessed = [' '.join([tok.text for tok in nlp.tokenizer(line)]) for line in tqdm(preprocessed)]

with open(os.path.join(data_dirpath, 'nyt_preprocessed.txt'), 'w') as f:
    for line in preprocessed:
        f.write(line + '\n')
