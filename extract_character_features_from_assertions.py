import os
from tqdm import tqdm
import json
import spacy
nlp = spacy.load('en', disable=['ner'])

def load_character_assertions(fandom, dataset_name):
    pipeline_output_path = f'/data/fanfiction_ao3/{fandom}/{dataset_name}/output_old/'
    char_assertions = {}
    assertions_dirpath = os.path.join(pipeline_output_path, 'assertion_extraction')

    # for fic, chars in tqdm(fic_chars.items()):
    for fname in tqdm(os.listdir(assertions_dirpath), ncols=50):
        fic_assertions_path = os.path.join(assertions_dirpath, fname)
    #     if not os.path.exists(fic_assertions_path):
    #         continue
            
        with open(fic_assertions_path) as f:
            char_assertions[fname[:-5]] = json.load(f)
            
    return char_assertions


def name_from_char(charname):
    name = ' '.join([part for part in charname.split('_') if len(part) > 0 and part[0].isupper()])
    if name.endswith(')'):
        name = name[:-1]
    return name


def normalize_names(text):
    text_split = text.split()
    
    for i, word in enumerate(text_split):
        if word.startswith('($_'):
            name = name_from_char(word)
            text_split[i-1] = name
            text_split[i] = ''

    return ' '.join(text_split)


def main():
    char_features = {}

    fandom = 'supernatural'
    dataset_name = 'complete_en_1k-50k'

    # Load character assertions
    char_assertions = load_character_assertions(fandom, dataset_name)
    print(f"Assertions from {len(char_assertions)} files loaded.")
    pipeline_output_path = f'/data/fanfiction_ao3/{fandom}/{dataset_name}/output_old/'

    output_dirpath = os.path.join(pipeline_output_path, 'char_features')
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    # Process assertions
    # for fic in list(char_assertions.keys())[:1]:
    for fic in tqdm(char_assertions, ncols=50):

        # Check if already processed
        if os.path.exists(os.path.join(output_dirpath, f'{fic}.json')):
            continue

        if not fic in char_features:
            char_features[fic] = {}

        for char in char_assertions[fic]:
            name = name_from_char(char)
            assertions = ' '.join(char_assertions[fic][char])
            
            # Replace pronouns with coref'ed names
            assertions = normalize_names(assertions)
            
            # postag and parse
            annotated = nlp(assertions)
            
            # Verbs where character was the subject
            verbs_subj = [tok.head.text.lower() for tok in annotated if tok.text == name and (tok.dep_=='nsubj' or tok.dep_=='agent')]
            verbs_subj
            
            # Verbs where character was the object
            verbs_obj = [tok.head.text.lower() for tok in annotated if tok.text == name and (tok.dep_=='dobj' or tok.dep_=='nsubjpass' or tok.dep_=='dative' or tok.dep_=='pobj')]
            
            # Adjectives that describe the character
            adjs = [tok.text.lower() for tok in annotated if tok.head.text == name and (tok.dep_=='amod' or tok.dep_=='appos' or tok.dep_=='nsubj' or tok.dep_=='nmod')] \
            + [tok.text.lower() for tok in annotated if tok.dep_=='attr' and (tok.head.text=='is' or tok.head.text=='was') and name in [c.text for c in tok.head.children]]
            
            if not name in char_features[fic]:
                char_features[fic][name] = []
                
            char_features[fic][name].extend(verbs_subj + verbs_obj + adjs)
            
        # Save out
        with open(os.path.join(output_dirpath, f'{fic}.json'), 'w') as f:
            json.dump(char_features[fic], f)

if __name__ == '__main__': main()
