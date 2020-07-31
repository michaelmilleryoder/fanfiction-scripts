#!/usr/bin/env python
# coding: utf-8
import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm as tqdm
import pdb


# # Split HP canon into chapters, sentences, save to CSVs

canon_dirpath = '/projects/fanfic_embeddings/from_James/canon_data/harry_potter_tokenized'
chap_pattern = r'Chapter \d+'
book_chap_titles = []
outlines = []

for book_id in range(1,8):
    print(book_id)
    book_fpath = f'/projects/fanfic_embeddings/from_James/canon_data/harry_potter_tokenized/Harry_Potter_{book_id}.tokenized.txt'
    with open(os.path.join(book_fpath)) as f:
        text = f.read().splitlines()
    book_title = text[0][1:-1]
    text = text[1:]

    # Build dataframe of book, chapter titles
    chap_indices = [i for i in range(len(text)) if re.match(chap_pattern, text[i])]
    book_chap_titles += [[book_id, book_title, i+1, text[ind+1]] for i, ind in enumerate(chap_indices)]

    # Split into chapters
    chap_texts = []

    current_index = 0
    for ind in chap_indices:
        chap_texts.append(text[current_index:ind])
        current_index = ind+2 # skip chapter title
        
    chap_texts = chap_texts[1:]
    print(f'{len(chap_texts)} chapters found')


    # Divide into sentences, words
    prev_sent_len = 0
    for c, chap in enumerate(tqdm(chap_texts)):
        for p, para in enumerate(chap):
            sentence_id = 0
            for s, sent in enumerate(sent_tokenize(para)):
                if len(sent.split()) == 1 and sent[0] == '”': # 1 token, attach to previous sentence
                    outlines.append([book_id, c+1, p+1, sentence_id, prev_sent_len+1, sent.split()[0]])
                else:
                    sentence_id += 1
                    for t, tok in enumerate(sent.split()):
                        outlines.append([book_id, c+1, p+1, sentence_id, t+1, tok])
                    prev_sent_len = t + 1

    # Save out token file
    tok_df = pd.DataFrame(outlines, columns=['book_id', 'chapter_id', 'paragraph_id', 'sentence_id', 'word_id', 'word'])
    tok_df.to_csv(f'/home/mamille2/storyq/harrypotter_book{book_id}_words.csv', index=False)


# Save out 
chap_df = pd.DataFrame(book_chap_titles, columns=['book_id', 'book_title', 'chapter_id', 'chapter_title'])
chap_df.to_csv('/home/mamille2/storyq/harrypotter_book_chapter_titles.csv', index=False)

#tok_df = pd.DataFrame(outlines, columns=['book_id', 'chapter_id', 'paragraph_id', 'sentence_id', 'word_id', 'word'])
#tok_df.to_csv('/home/mamille2/storyq/harrypotter_words.csv', index=False)
print("Saved tokens and book, chapter information")
