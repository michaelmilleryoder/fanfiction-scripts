"""
    Create character feature dataset for STM
"""

import os
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
import re
from character_annotator import CharacterAnnotator

import pdb

class FandomDatasetCreator():
    
    def __init__(self, fandom, json_features_dirpath, outpath):
        self.fandom = fandom
        self.json_features_dirpath = json_features_dirpath
        self.outpath = outpath
        self.character_annotator = CharacterAnnotator(fandom)

    def annotate_relationship_types(self):
        """ Annotate relationship types in self.data """
        # Load fic metadata
        metadata_fpath = f'/data/fanfiction_ao3/{self.fandom}/complete_en_1k-50k/metadata.csv'
        metadata = pd.read_csv(metadata_fpath)

        merged = pd.merge(self.data, metadata, on=['fic_id'])
        char_in_rel = []
        char_in_rel_type = []
        for char, rel in zip(merged['character_original'], merged['relationship']):
            x, y = self.character_annotator.char_in_relationship(char, rel)
            char_in_rel.append(x)
            char_in_rel_type.append(y)
        #merged['character_in_relationship'], merged['character_in_relationship_type'] = list(zip(*[self.character_annotator.char_in_relationship(tup[0], tup[1]) for tup in zip(merged['character_original'], merged['relationship'])]))
        merged['character_in_relationship'] = char_in_rel
        merged['character_in_relationship_type'] = char_in_rel_type
        #print(merged['character_in_relationship'].sum())
        merged['character_in_relationship_type'].value_counts()
        #merged.drop(columns=['char_in_relationship'], inplace=True)
        self.data = merged

    def assemble_dataset(self):
        # Aggregate character features from JSONs
        header = ['fic_id', 'character_original', 'character_canonical', 'character_gender', 'character_features']
        outlines = []
        self.character_annotator.character_name_parts() # to exclude in features
        print("Loading character genders...")
        character_gender = self.character_annotator.character_gendermap()
        print("Assembling character features...")
        #for fname in tqdm(os.listdir(self.json_features_dirpath), ncols=50):
        #    fic_id = fname.split('.')[0]
        #    with open(os.path.join(self.json_features_dirpath, fname)) as f:
        #        char_features = json.load(f)
        #    for char in char_features:
        #        canonical_char = self.character_annotator.canonicalize(char)
        #        if not canonical_char in character_gender: continue
        #        gender = character_gender[canonical_char]
        #        feats = ' '.join([feat for feat in char_features[char] if not feat in self.character_annotator.character_name_parts])
        #        if len(feats) < 1: continue
        #        outlines.append([fic_id, char, canonical_char, gender, feats])
        #self.data = pd.DataFrame(outlines, columns=header)
        #self.data.to_csv(self.outpath, index=False)
        self.data = pd.read_csv(self.outpath)

        # Annotate fic relationship type
        print("Annotating relationship types...")
        self.annotate_relationship_types()
                    
        # Save out
        self.data.to_csv(self.outpath, index=False)
        print(f"Saved assembled character feature data from {len(self.data['fic_id'].unique())} fics")


def main():
    fandoms = [
        'harrypotter',
        'supernatural'
    ]
    data = {} # character features, to be merged

    for fandom in fandoms:
        print(f"Building or loading dataset from {fandom}...")
        json_features_dirpath = f'/data/fanfiction_ao3/{fandom}/complete_en_1k-50k/output_old/char_features'
        output_path = f'/data/fanfiction_ao3/{fandom}/complete_en_1k-50k/output/character_features.csv'
        if os.path.exists(output_path):
            print("\tAlready exists") 
            data[fandom] = pd.read_csv(output_path)
        else:
            # Load character features JSON files, merge with gender, output CSV
            creator = FandomDatasetCreator(fandom_name, json_features_dirpath, output_path)
            creator.assemble_dataset()
            data[fandom] = creator.data

    print("Merging datasets...")
    merged = pd.concat([data[fandom] for fandom in fandoms], keys=fandoms, names=['fandom_dataset', 'old_index'])
    merged.reset_index(inplace=True, level='fandom_dataset')
    merged_outpath = f'/data/fanfiction_ao3/character_features_{len(fandoms)}fandoms.csv'
    merged.to_csv(merged_outpath, index=False)
    print(f"\tSaved to {merged_outpath}")


if __name__ == '__main__':
    main()
