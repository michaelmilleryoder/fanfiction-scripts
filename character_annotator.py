import json
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import urllib.error
import os
import re
from collections import Counter
import pdb


def count_gendered_pronouns(soup, n_paras=3):
    """ Count gendered pronouns in the first n_paras of a soup html representation. """
    paras = [p.get_text() for p in soup.find_all('p')[:n_paras]]
    male_pronouns = [r'\bhe\b', r'\bhim\b', r'\bhis\b']
    female_pronouns = [r'\bshe\b', r'\bher\b', r'\bhers\b']
    neutral_pronouns = [r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\btheirs\b']
    male_pronoun_count = sum([sum([len(re.findall(pronoun, para)) for pronoun in male_pronouns]) for para in paras])
    female_pronoun_count = sum([sum([len(re.findall(pronoun, para)) for pronoun in female_pronouns]) for para in paras])
    neutral_pronoun_count = sum([sum([len(re.findall(pronoun, para)) for pronoun in neutral_pronouns]) for para in paras])
    return {'male': male_pronoun_count, 
            'female': female_pronoun_count,
            'neutral': neutral_pronoun_count}


class CharacterAnnotator():
    """ Handles lists of characters from AO3 and annotates them for genders 
        based on external resource (Wikia FANDOM) """

    def __init__(self, fandom):
        self.fandom = fandom
        self.remove = { 'harrypotter': [
                'Hogwarts', 'Slytherin', 'Gryffindor', 'Order', 'The',
                 'The Slytherin',
                  'The Headmaster',
                  'Weasley',
                  'Weasleys',
                  'Hufflepuff',
                 ],
                'supernatural': []
                }
        self.name_transform = { # Might then miss when stories use these terms, though (same with canonicalize): might want to spread gender info afterward
                'harrypotter': {
            'Potter': 'Harry Potter',
            'Mr. Potter': 'Harry Potter',
            'Mr Potter': 'Harry Potter',
            'The Harry': 'Harry Potter',
            'Mr Weasley': 'Ron Weasley',
            'Tom': 'Tom Riddle',
            'Black': 'Sirius Black',
            'The Draco': 'Draco Malfoy',
            'The Dark Lord': 'Voldemort',
            'The Dark Lord Voldemort': 'Voldemort',
            'Dark Lord': 'Voldemort',
            'Lord': 'Voldemort',
            'Malfoy': 'Draco Malfoy',
            'James': 'James Potter I',
            'James Potter': 'James Potter I',
            'Lily': 'Lily J. Potter',
            'Albus': 'Albus Dumbledore',
            'Teddy': 'Teddy Lupin',
            'Newt': 'Newton Scamander',
            'Rose': 'Rose Granger-Weasley',
            'Lily Potter': 'Lily J. Potter',
            'Regulus': 'Regulus Black I',
            'Mrs Weasley': 'Molly Weasley',
            },
            'supernatural': {}
        }
        self.wikia_urls = {  # self.fandom: wikia_name
            'harrypotter': 'harrypotter',
            'supernatural': 'supernatural'
        }

    def canonical_character_list(self):
        """ Load canonical character list, or build it, save in self.canonical_characters """
        char_list_fpath = f'/data/fanfiction_ao3/{self.fandom}/canonical_characters.txt'
        with open(char_list_fpath) as f:
            canonical_characters = f.read().splitlines()
        extra = ['Dobby']
        canonical_characters += extra
        canonical_characters = [c for c in canonical_characters if not c in self.remove[self.fandom]]
        self.canonical_characters = canonical_characters

    def character_gendermap(self):
        """ Load or build dictionary from canonical character names to genders """
        char_gender_fpath = f'/data/fanfiction_ao3/{self.fandom}/character_genders.json'        
        if os.path.exists(char_gender_fpath):
            # Load character gender dictionary
            with open(char_gender_fpath, 'r') as f:
                character_gender = json.load(f)
        else:
            character_gender = self.build_character_gendermap()

        self.character_gender = character_gender
        return character_gender

    def build_character_gendermap(self, force_load=False):
        """ Annotate character gender """
        tmp_dirpath = f'/data/fanfiction_ao3/{self.fandom}/tmp'
        charnames_fpath = os.path.join(tmp_dirpath, 'character_names.json')

        # Load all character names
        if not force_load and os.path.exists(charnames_fpath):
            with open(charnames_fpath, 'r') as f:
                character_names = json.load(f)
        else:
            char_features_dirpath = f'/data/fanfiction_ao3/{self.fandom}/complete_en_1k-50k/output_old/char_features'
            character_names = []
            for fname in tqdm(sorted(os.listdir(char_features_dirpath))):
                with open(os.path.join(char_features_dirpath, fname)) as f:
                    char_features = json.load(f)
                character_names.extend(list(char_features.keys()))
        if not os.path.exists(tmp_dirpath):
            os.mkdir(tmp_dirpath)
        with open(charnames_fpath, 'w') as f:
            json.dump(character_names, f)

        character_names_ctr = Counter(character_names)
        print(len(character_names_ctr))

        # Refine characters
        characters_ctr_filtered = Counter({name: count for name,count in character_names_ctr.items() if name != '' and count > 1})

        # Load canonical list of characters
        if not hasattr(self, 'canonical_characters'):
            self.canonical_character_list()

        # Remove characters who don't have matches with canonical characters
        # Filter, canonicalize names
        characters_ctr_matches = Counter()
        for name, count in characters_ctr_filtered.items():
            new_name = self.canonicalize(name)
            
            # Remove names
            if len(new_name) <= 1 or new_name in self.remove[self.fandom]:
                continue
                
            # Transform names
            if new_name in self.name_transform[self.fandom]:
                new_name = self.name_transform[self.fandom[new_name]]
            if new_name in characters_ctr_matches:
                characters_ctr_matches[new_name] += count
            else:
                characters_ctr_matches[new_name] = count

        character_gender = {} # Keys are character names from character features
        characters_no_genderbox = []
        characters_http_error = []

        # Get gender from wikia page
        for character in tqdm([name for name, count in characters_ctr_matches.most_common()], ncols=50):
            if character in character_gender: continue
            if character in characters_no_genderbox: continue
            if character in characters_http_error: continue
            url_base = f'https://{self.wikia_urls[self.fandom]}.fandom.com/wiki/'
            try:
                html_str = urllib.request.urlopen(url_base + character.replace(' ', '_')).read()
            except urllib.error.HTTPError as e:
                characters_http_error.append(character)
                tqdm.write(f'Character {character} HTTP error')
                
            soup = BeautifulSoup(html_str, 'html.parser')
            if soup is None:
                print(html_str)
                break

            # Check for disambiguation page
            if 'Disambiguation' in soup.find('div', {'class': 'page-header__categories-links'}).get_text():
                tqdm.write(f"Character {character} can't be disambiguated")
                continue

            # Assign gender
            # div = list(soup.find('div', {'data-source':"gender"}).children)
            # soup.find('div', {'data-source':"gender"}).text
            genderbox = soup.find('div', {'data-source':"gender"})
            if genderbox is None:
                characters_no_genderbox.append(character)
                #if character == 'Mary': pdb.set_trace()
                pronoun_counts = count_gendered_pronouns(soup)
                gender = max(pronoun_counts, key=pronoun_counts.get) 
                character_gender[character] = gender
                tqdm.write(f'Character {character} has no gender box, set by pronouns')
            else:
                gender = re.match(r'\w+', genderbox.find('div').text.lower()).group()
                pdb.set_trace()
                time.sleep(.5)
                character_gender[character] = gender
            
        # See how many characters, uses have annotated gender
        print(f"{len(character_gender)}/{len(characters_ctr_matches)} ({len(character_gender)/len(characters_ctr_matches): .1%}) characters annotated for gender")
        labeled_uses = sum([characters_ctr_matches[name] for name in character_gender])
        total_uses = sum(characters_ctr_matches.values())
        print(f'{labeled_uses/total_uses: .1%} mentions of characters annotated for gender')

        # Save character gender json
        print("Saving character gender mapping...")
        with open(f'/data/fanfiction_ao3/{self.fandom}/character_genders.json', 'w') as f:
            json.dump(character_gender, f)

    def canonicalize(self, name):
        """ Return canonical name from a name """
        name_parts = name.split()
        new_name_parts = []
        if not hasattr(self, 'character_name_parts'):
            self.character_name_parts()
        for name_part in name_parts:
            if name_part.lower() in self.character_name_parts:
                new_name_parts.append(name_part)
        new_name = ' '.join(new_name_parts)
        if new_name in self.remove[self.fandom]:
            new_name = ''
        if new_name in self.name_transform[self.fandom]:
            new_name = self.name_transform[self.fandom][new_name]
        return new_name

    def character_name_parts(self):
        """ Returns parts of character names, to skip in feature extraction.
            Also saves character name parts to self.character_name_parts
        """

        # Load canonical list of characters
        if not hasattr(self, 'canonical_characters'):
            self.canonical_character_list()
        canonical_character_name_parts = set([part for name in self.canonical_characters for part in name.split()])
        exclude = set(['The'])
        canonical_character_name_parts -= exclude
        name_parts_lower = [c.lower() for c in canonical_character_name_parts]
        self.character_name_parts = name_parts_lower
        return name_parts_lower

    def char_in_relationship(self, char, rel):
        # TODO: consider relationships with more than 2 characters differently
        
        match = False
        rel_type = 'character_not_in_relationship'
        
        char_parts = set([c.lower() for c in char.split()])
        
        # Is the character in a relationship?
        for relationship in eval(rel):
            if not '/' in relationship: continue # isn't romantic
            rel_parts = set([c.lower() for c in re.split(r'[\/ ]', relationship)])
            if len(char_parts.intersection(rel_parts)) > 0:
                match = True
                
                # Determine other character in relationship and their gender
                rel_chars = relationship.split('/')
                other_char = ''
                for rel_char in rel_chars:
                    if not any([char_part in rel_char.lower() for char_part in char_parts]):
                        other_char = self.canonicalize(rel_char)
                        
                # Determine gender of char, other char in relationship
                char_gender = self.character_gender[self.canonicalize(char)]
                if not other_char in self.character_gender: # other char gender unknown
                    rel_type = 'unknown'
                else:
                    other_char_gender = self.character_gender[other_char]
                
                    if char_gender != other_char_gender:
                        rel_type = 'straight'

                    else:
                        rel_type = 'queer'
        
                break
        return (match, rel_type)
