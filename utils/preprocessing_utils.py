import string
import pickle
import os.path
import re
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer


def convert_tokens_to_padded_sequence(X_tokenized, word_embedding_mapping, max_length):
    """ Transforms the texts in the collection to sequences of indices.
    It also pads or truncates the texts to the specified length.

    Keyword arguments:
    X_tokenized -- the tokenized data to be transformed
    word_embedding_mapping -- the mapping from a token to the index in the embedding matrix
    max_length -- the resulting length of each text
    """
    X_sequence = X_tokenized.apply(lambda comment: [word_embedding_mapping[token] for token in comment])
    X_result = pad_sequences(X_sequence, max_length)
    return X_result


def tokenize_sentences(X, reduce_len=False):
    """ Tokenizes the texts in the given series.

    Keyword arguments:
    X -- the series to be tokenized
    """
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=reduce_len)
    return X.apply(lambda x: tknzr.tokenize(x))


def remove_punctuation(X):
    """ Removes all punctuation from the dataset
    
    Arguments:
        X {DataFrame} -- The dataset to be transformed
    
    Returns:
        DataFrame -- The dataset with all punctuation removed
    """
    pattern = '[^A-Za-z0-9\\s -]'
    return X.apply(lambda x: re.sub(pattern, '', x))


def remove_punctuation_weak(X):
    """ Removes all punctuation except ,.!? from the dataset
    
    Arguments:
        X {DataFrame} -- The dataset to be transformed
    
    Returns:
        DataFrame -- The dataset with all punctuation except ,.!? removed
    """
    pattern = '[^A-Za-z0-9\\s,!?\\. -]'
    return X.apply(lambda x: re.sub(pattern, '', x))

