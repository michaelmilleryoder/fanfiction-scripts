""" Superclass data structure for holding a predicted or annotate span of text """

import re
import itertools
from string import punctuation
from collections import Counter
import pdb


def match_links(links, ref_links):
    """ Returns matching links between 2 sets of links.
        Links are tuples of the form ((location0, location1))
    """
    common_links = set()
    for location0, location1 in links:
         if (location0, location1) in ref_links or \
            (location1, location0) in ref_links:
            common_links.add((location0, location1))
    return common_links


def match_spans(predicted_spans, gold_spans, exact=True):
    """ Match AnnotatedSpan objects just on extracted spans (ignoring annotations).
        Args:
            predicted_quotes: AnnotatedSpan objects predicted
            gold_quotes: AnnotatedSpan objects annotated as gold truth
            exact: whether an exact match on token IDs is necessary.
                For the FanfictionNLP pipeline, this should be the case.
                For baseline systems that might have different tokenization;
                this can be set to False to relax that constraint.
    """

    matched_gold = []
    matched_predicted = []
    false_positives = []
    false_negatives = []

    matched = [(predicted, gold) for predicted, gold in itertools.product(predicted_spans, gold_spans) if gold.span_matches(predicted, exact=exact)]
    if len(matched) == 0:
        matched_predicted, matched_gold = [], []
    else:
        matched_predicted, matched_gold = list(zip(*matched))

    false_positives = [predicted for predicted in predicted_spans if not predicted in matched_predicted]
    false_negatives = [gold for gold in gold_spans if not gold in matched_gold]

    return list(matched_predicted), list(matched_gold), false_positives, false_negatives


def match_annotated_spans(predicted_spans, gold_spans, matched=False, incorrect_extractions=[]):
    """ Match AnnotatedSpan objects entirely, including annotation.
        If extractions don't match, counts it as a mismatched annotation.
        Returns matched_attributions, mismatched_attributions
    """

    correct_attributions = []

    # Check extractions
    if not matched:
        matched_predicted, matched_gold, incorrect_extractions, _ = match_spans(predicted_spans, gold_spans)
    else:
        matched_gold, matched_predicted = gold_spans, predicted_spans

    incorrect_attributions = incorrect_extractions.copy()

    # Find matched gold spans
    for pred_span, gold_span in zip(matched_predicted, matched_gold):
    
        # Check attribution
        if characters_match(pred_span.annotation, gold_span.annotation):
            correct_attributions.append((pred_span, gold_span))
        else:
            incorrect_attributions.append((pred_span, gold_span))

    return correct_attributions, incorrect_attributions


def annotation_labels(attributions1, attributions2, matched=False, mismatched_extractions1=[], mismatched_extractions2=[]):
    """ Get lists of labels (including null) from 2 lists of AnnotatedSpans
        Returns matched_attributions, mismatched_attributions, mismatched_extractions where each is a list of tuples (annotated_span1, annotated_span2)
    """
    # TODO: merge with match_annotated_spans?

    matched_attributions = []
    mismatched_attributions = []
    mismatched_extractions = []

    # Check extractions
    if not matched:
        matched1, matched2, spans_1not2, spans_2not1 = match_spans(attributions1, attributions2)
    else:
        matched1, matched2 = attributions1, attributions2
        spans_1not2 = mismatched_extractions1
        spans_2not1 = mismatched_extractions2

    # Find matched gold spans
    for span1, span2 in zip(matched1, matched2):
    
        # Check attribution
        if characters_match(span1.annotation, span2.annotation):
            matched_attributions.append((span1, span2))
        else:
            mismatched_attributions.append((span1, span2))

    for span in spans_1not2:
        mismatched_extractions.append((span, span.null_span()))
    for span in spans_2not1:
        mismatched_extractions.append((span.null_span(), span))

    return matched_attributions, mismatched_attributions, mismatched_extractions


def normalize_annotations_to_id(annotations1, annotations2):
    """ Returns a dictionary of character name to the same ID if the characters match.
        Pass the same annotations twice to reduce a single set of annotations.
    """
    char2id = {}
    current_id = 0
    labels1 = {span.annotation for span in annotations1}
    labels2 = {span.annotation for span in annotations2}

    for label1 in labels1:
        char2id[label1] = current_id
        for label2 in labels2:
            if characters_match(label1, label2):
                char2id[label2] = current_id
                break
        current_id += 1

    # any unmatched characters
    for label2 in labels2:
        if not label2 in char2id:
            char2id[label2] = current_id
            current_id += 1

    return char2id


def normalize_annotations_to_name(annotations):
    """ Returns annotations with character names normalized such that 
        no character name matches another.
        Keeps the first seen instance of character names that match each other.
    """
    # TODO: Merge with normalize_annotations_to_id
    normalized = [] # list of normalized names
    normalized_annotations = []
    for span in annotations:
        for name in normalized:
            if span.annotation == name: # plain match, for speed
                normalized_annotations.append(span)
                break
            if characters_match(span.annotation, name):
                normalized_annotations.append(span.change_annotation(name))
                break
        else:
            normalized.append(span.annotation)
            normalized_annotations.append(span)

    return normalized_annotations


def group_annotations(spans):
    """ Group annotations by annotation
        Returns dictionary of {annotation: [AnnotatedSpan, ...]}
    """

    clusters = {}
    for span in spans:
        if not span.annotation in clusters:
            clusters[span.annotation] = []
        clusters[span.annotation].append(span)

    return clusters


def group_annotations_para(spans):
    """ Group annotations by (chap_id, para_id)
        Returns dictionary of {(chap_id, para_id): [AnnotatedSpan, ...]}
    """

    clusters = {}
    for span in spans:
        key = (span.chap_id, span.para_id)
        if not key in clusters:
            clusters[key] = []
        clusters[key].append(span)

    return clusters


def characters_match(char1, char2):
    """ Returns True if 2 character names match closely enough.
        First splits character names into parts by underscores or spaces.
        Matches either if:
            * Any part matches and either name has only 1 part (Potter and Harry Potter, e.g.)
            * The number of part matches is higher than half of unique name parts between the 2 characters
    """
    honorifics = ['ms.', 'ms',
                    'mr.', 'mr',
                    'mrs.', 'mrs',
                    'uncle', 'aunt',
                    'dear', 'sir', "ma'am"
                ]
    char1_processed = re.sub(r'[\(\),]', '', char1)
    char1_parts = [part for part in re.split(r'[ _]', char1_processed.lower()) if not part in honorifics] 
    char2_processed = re.sub(r'[\(\),]', '', char2)
    char2_parts = [part for part in re.split(r'[ _]', char2_processed.lower()) if not part in honorifics]
    
    # Count number of part matches
    n_parts_match = 0
    for part1 in char1_parts:
        for part2 in char2_parts:
            #if part1 == part2 and len(char1_parts)==1 or len(char2_parts)==1:
            if part1 == part2:
                n_parts_match += 1

    # Determine match
    not_surnames = ['male', 'female']
    if n_parts_match == 1 and (len(char1_parts) == 1 or len(char2_parts) == 1) and not any([w in char1_parts for w in not_surnames]) and not any([w in char2_parts for w in not_surnames]):
        match = True
    elif n_parts_match > len(set(char1_parts + char2_parts))/2:
        match = True
    else:
        match = False

    return match


def spans_union(spans_list, exact=True, attribution_conflicts='remove'):
    """ Returns the union of all spans, handling attributions 
        Args:
            spans_list: list of lists of AnnotatedSpan objects
            exact: whether the match should be exact
            attribution_conflicts: what do with mismatching attributions {'remove', 'ignore'}
    """

    all_spans = spans_list[0].copy()
    for spans in spans_list[1:]:
        new_spans = []
        mismatched_attributions = []
        for span in spans:
            for existing_span in all_spans:
                if span.span_matches(existing_span, exact=exact):
                    if not characters_match(span.annotation, existing_span.annotation):
                       mismatched_attributions.append(span)
                    break # everything matches, don't add as a unique span
            else:
                new_spans.append(span)
        all_spans += new_spans

    if attribution_conflicts == 'remove':
        all_spans = [span for span in all_spans if not any([span.span_matches(conflict_span) for conflict_span in mismatched_attributions])]

    return all_spans


def all_characters(spans):
    """ Returns all unique annotations (characters) from a list of AnnotatedSpan objects """
    return sorted(set(span.annotation for span in spans))


def canonical_character_name(names):
    """ Returns a chosen canonical character name.
        Args:
            names: a Counter of name variants
    """

    if len(names) == 1:
        return list(names.keys())[0]

    # Remove stopwords
    # Choose most frequent name that has a capital letter
    processed_names = Counter()
    stops = ['he', 'him', 'his', 'himself',
             'she', 'her', 'hers', 'herself',
             'they', 'them', 'their', 'theirs',
             'i', 'me', 'my', 'mine',
             'we', 'us', 'our', 'ours',
             'you', 'your', 'yours', 'yourself',
            'mr.', 'mr', 'ms', 'ms.', 'miss', 'miss.', 'mrs', 'mrs.',
            'sir',
            ]
    for name, count in names.items():
        if name.lower() not in stops:
            processed_names[name] = count
    if len(processed_names) == 0:
        processed_names = names

    capitalized = Counter({
            name: count for name, count in processed_names.items() \
            if any(letter.isupper() for letter in name)})
    if len(capitalized) == 0:
        capitalized = processed_names

    return capitalized.most_common(1)[0][0]


class AnnotatedSpan():

    def __init__(self, 
            chap_id=None, 
            para_id=None, 
            start_token_id=None, 
            end_token_id=None, 
            annotation=None, # speaker for quotes, or character for character mentions
            text=''):
        self.chap_id = chap_id # starts with 1, just like annotations
        self.para_id = para_id # starts with 1, just like annotations
        self.start_token_id = start_token_id # starts over every paragraph, starts with 1 just like annotations
        self.end_token_id = end_token_id
        self.annotation = annotation
        self.text = text
        self.text_tokens = None

    def __repr__(self):
        return f"{self.chap_id}.{self.para_id}.{self.start_token_id}-{self.end_token_id},annotation={self.annotation}"

    def readable_span(self):
        if self.start_token_id == self.end_token_id:
            outstring = f"{self.chap_id}.{self.para_id}.{self.start_token_id}"
        else:
            outstring = f"{self.chap_id}.{self.para_id}.{self.start_token_id}-{self.end_token_id}"
        return outstring

    def null_span(self):
        """ Returns an identical span but with a NULL annotation"""
        return AnnotatedSpan(
            chap_id=self.chap_id,
            para_id=self.para_id, 
            start_token_id=self.start_token_id, 
            end_token_id=self.end_token_id, 
            annotation='NULL')

    def change_annotation(self, new_annotation):
        """ Returns an identical span but annotated with new_annotation"""
        return AnnotatedSpan(
            chap_id=self.chap_id,
            para_id=self.para_id, 
            start_token_id=self.start_token_id, 
            end_token_id=self.end_token_id, 
            annotation=new_annotation)

    def get_location(self):
        return (self.chap_id, self.para_id, self.start_token_id, self.end_token_id)

    def span_matches(self, other_span, exact=True):
        """ Check if the extracted span matches another span.
            Ignores attribution.
            Args:
                exact: whether an exact match on token IDs is necessary.
                    Otherwise matches if is in the same paragraph, 
                    has very similar text and beginning and start points 
                    occur within a small window of the other span.
        """

        if not (self.chap_id == other_span.chap_id and \
            self.para_id == other_span.para_id):
            return False

        if exact:
            return self.span_endpoints_align(other_span, exact=exact)
        return self.span_text_matches(other_span) and \
                self.span_endpoints_align(other_span, exact=exact)

    def span_endpoints_align(self, other_span, exact=True):
        """ Returns whether span endpoints are within a small window
            of each other (or exact if specified).
        """

        if exact:
            return self.start_token_id == other_span.start_token_id and \
                self.end_token_id == other_span.end_token_id
        window_size = 3
        return abs(self.start_token_id - other_span.start_token_id) <= window_size and abs(self.end_token_id - other_span.end_token_id) <= window_size

    def span_text_matches(self, other_span):
        word_match_threshold = .5

        if not hasattr(self, 'text_tokens'):
            self.preprocess_text()
        if not hasattr(other_span, 'text_tokens'):
            other_span.preprocess_text()
        
        # Measure unique word overlap
        n_matches = len(self.text_tokens.intersection(other_span.text_tokens))

        # Check for edge cases
        if len(other_span.text_tokens) == 0:
            return self.text_tokens < 4 # Probably just the name of a character

        return (n_matches/len(other_span.text_tokens)) >= word_match_threshold

    def preprocess_text(self):
        """ Creates a set of lowercased unique tokens from a quote's text.
            Saves to self.text_tokens
        """

        # Remove ccc_ business
        processed_quote = re.sub(r'ccc_.*?_ccc', '', self.text)

        # Remove punctuation, lowercase
        stops = list(punctuation) + ['”', '“']
        processed_quote = ''.join([c for c in processed_quote.lower() if not c in stops])

        # Replace whitespace with spaces
        #processed_quote = re.sub(r'\s+', ' ', processed_quote)
        
        # Extract unique words
        processed_words = set(processed_quote.strip().split())

        self.text_tokens = processed_words
