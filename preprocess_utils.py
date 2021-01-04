""" A combination of utilities for preprocessing fanfiction scraped from AO3.
    @author Michael Miller Yoder <yoder@cs.cmu.edu>
    @date 2021
"""

def load_metadata_file(fpath):
    data = pd.read_csv(fpath)
    # Resolve bytestring issue (should be resolved with newer scrapes)
    data = data.applymap(remove_bytestring)
    # Make sure fic_id is an int column
    data['fic_id'] = data['fic_id'].astype(int)
    # Remove duplications
    data = data.drop_duplicates('fic_id')
    return data


def copy_fic(tuple_args):
    fic_id, input_dirpath, output_dirpath = tuple_args
    inpath = os.path.join(input_dirpath, f'{fic_id}.csv')
    outpath = os.path.join(output_dirpath, f'{fic_id}.csv')
    shutil.copy(inpath, outpath)


def just_copy_fics(fic_list, input_dirpath, output_dirpath, num_cores=1):
    """ Simply copy story files (fics) to output directory.
        No merging of chapter files into story files.
        Args:
            num_cores: number of cores (>1 multiprocessing). However,
                    doesn't appear to speed things up unless the server
                    is running slow I/O operations.
    """
    fics_with_no_data = []
    fic_ids_to_write = []
    tqdm.write(f"\t{len(fic_list)} fics found to copy")

    # Find any missing data
    for fic_id in fic_list:
        inpath = os.path.join(input_dirpath, f'{fic_id}.csv')
        if not os.path.exists(inpath):
            fics_with_no_data.append(fic_id)
        else:
            fic_ids_to_write.append(fic_id)

    # Multiprocessing
    if num_cores > 2:
        with Pool(num_cores) as p:
            list(tqdm(p.imap(copy_fic, zip(
                fic_ids_to_write,
                itertools.repeat(input_dirpath),
                itertools.repeat(output_dirpath)
                )), total=len(fic_ids_to_write), ncols=70))
    else:
        for fic_id in tqdm(fic_list, ncols=70):
            inpath = os.path.join(input_dirpath, f'{fic_id}.csv')
            outpath = os.path.join(output_dirpath, f'{fic_id}.csv')
            shutil.copy(inpath, outpath)

    tqdm.write(f"\t{len(fics_with_no_data)} fics with metadata but no story data")

    return fics_with_no_data


def remove_bytestring(value):
    """ Remove issues with b' or b" prepended to strings with unicode encoding
        run with Python 3
    """
    new_value = value
    if isinstance(value, str):
        if value.startswith("b'") or value.startswith('b"'):
            new_value = value[2:-1]
    return new_value


def tokenize(text):
    if not isinstance(text, str):
        return ""
    else:
        return ' '.join([tok.text for tok in NLP.tokenizer(text)]).strip()


def tokenize_fic(fpath, force=True):
    """ Tokenize a fic.
        Args:
            fpath: Path to a CSV file with fic data. The column "text" will be tokenized and a column "text_tokenized" added.
            force: Whether to force new tokenization if the CSV already contains a "text_tokenized" column (default=False)
    """
    
    fic = pd.read_csv(fpath)
    fic = fic.applymap(remove_bytestring) # remove any bytestring issue
    if 'text_tokenized' in fic.columns and not force:
        return
    fic['text_tokenized'] = fic['text'].map(tokenize)
    fic.to_csv(fpath, index=False)


