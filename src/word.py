import torch

def split(string: str):
    '''
    Splits the given string into words.
    '''

    words = []
    curr_word = ""

    for c in string:
        if not c.isalpha():
            if len(curr_word) == 0:
                continue
            curr_word.lower()
            words.append(curr_word)
            curr_word = ""

        curr_word += c

    return words

def build_matrix(dict, words):
    '''
    Takes a dictionnary and a list of words, then produces a sparse matrix representing it.
    Each word is represented by a column in the vector. Every elements of the column is zero,
    except the one corresponding to the index of the word in the dictionnary.
    '''

    dict_size = len(dict)
    words_count = len(words)

    indices = torch.zeros(size=(2, words_count))
    values = []
    for i, w in enumerate(words):
        word_index = dict[w]
        indices[0, i] = word_index
        indices[1, i] = i
        values.append(1.)

    mat = torch.sparse_coo_tensor(indices, values, size=(dict_size, words_count))
    return mat
