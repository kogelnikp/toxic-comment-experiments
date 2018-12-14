import numpy as np
from gensim.models import KeyedVectors


def load_glove_embeddings(path):
    """ Load pretrained glove embeddings.

    Keyword arguments:
    path -- the path to the pretrained embeddings
    """
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(path))
    all_embeddings = np.stack(embeddings_index.values())
    embedding_mean,embedding_std = all_embeddings.mean(), all_embeddings.std()
    return embeddings_index, embedding_mean, embedding_std

def load_word2vec_embeddings(path):
    """ Load pretrained word2vec embeddings.

    Keyword arguments:
    path -- the path to the pretrained embeddings
    """
    en_model = KeyedVectors.load_word2vec_format(path, binary=True)
    embeddings_index = en_model.wv
    all_embeddings = np.stack(embeddings_index.syn0)
    del en_model
    embedding_mean,embedding_std = all_embeddings.mean(), all_embeddings.std()
    return embeddings_index, embedding_mean, embedding_std

def load_fasttext_embeddings(path):
    """ Load pretrained fasttext embeddings.

    Keyword arguments:
    path -- the path to the pretrained embeddings
    """
    en_model = KeyedVectors.load_word2vec_format(path, binary=False)
    embeddings_index = en_model.wv
    all_embeddings = np.stack(embeddings_index.syn0)
    embedding_mean,embedding_std = all_embeddings.mean(), all_embeddings.std()
    del en_model
    return embeddings_index, embedding_mean, embedding_std



def create_initial_embedding_matrix(X_train_tokenized, X_test_tokenized, embedding_idx, embedding_mean, embedding_std, word_embedding_length, debug=False):
    """ Create initial embeddings weights to be used by the models.

    Arguments:
        X_train_tokenized -- tokenized training data
        X_test_tokenized -- tokenized test data
        embedding_idx -- dictionary of words and their corresponding embedding vectors
        embedding_mean -- mean of the embedding values
        embedding_std -- std of the embedding values
        word_embedding_length -- length of an embeddings vector
    
    Keyword arguments:
        debug -- print debug output (default: False)

    Returns:
        matrix -- initial embedding matrix
        dict -- mapping from token to index
    """
    all_tokens = get_all_tokens_in_dataset(X_train_tokenized, X_test_tokenized)
    initial_matrix = np.random.normal(embedding_mean, embedding_std, (len(all_tokens) + 1, word_embedding_length))
    initial_matrix[0, :] = np.zeros((1, word_embedding_length))

    word_matrix_mapping = {}

    count_tokens_in_pretrained_model = 0
    idx = 1
    for token in all_tokens:
        if token in embedding_idx:
            initial_matrix[idx] = embedding_idx[token]
            count_tokens_in_pretrained_model += 1
        word_matrix_mapping[token] = idx
        idx += 1
    
    if debug:
        print('Number of unique tokens: %d' % (len(all_tokens)))
        print('Number of tokens found in pretrained embeddings: %d' % (count_tokens_in_pretrained_model))
    
    return initial_matrix, word_matrix_mapping


def create_embeddings_mapping(X_train_tokenized, X_test_tokenized, debug=False):
    """create the mapping from each token to its index
    
    Arguments:
        X_train_tokenized {DataFrame} -- train set
        X_test_tokenized {DataFrame} -- test set
    
    Keyword Arguments:
        debug {bool} -- print debug output (default: {False})
    
    Returns:
        dict -- mapping from token to index
    """

    all_tokens = get_all_tokens_in_dataset(X_train_tokenized, X_test_tokenized)
    word_matrix_mapping = {}

    idx = 1
    for token in all_tokens:
        word_matrix_mapping[token] = idx
        idx += 1
    
    if debug:
        print('Number of unique tokens: %d' % (len(all_tokens)))

    return word_matrix_mapping


def get_all_tokens_in_dataset(X_train_tokenized, X_test_tokenized):
    """returns a list including all unique tokens in train and test set
    
    Arguments:
        X_train_tokenized {DataFrame} -- train set
        X_test_tokenized {DataFrame} -- test set
    
    Returns:
        list -- list of unique tokens
    """

    X_train_sublists = X_train_tokenized.values.flatten()
    X_train_tokens = set([item for sublist in X_train_sublists for item in sublist])    
    X_test_sublists = X_test_tokenized.values.flatten()
    X_test_tokens = set([item for sublist in X_test_sublists for item in sublist])
    return list(X_train_tokens | X_test_tokens)