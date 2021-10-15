def levenstein_distance(expected, result):
    if len(expected) == 0:
        return len(result)
    elif len(result) == 0:
        return len(expected)
    elif expected[0] == result[0]:
        return levenstein_distance(expected, result)
    else:
        a = levenstein_distance(expected[1:], result)
        b = levenstein_distance(expected, result[1:])
        c = levenstein_distance(expected[1:], result[1:])

        return 1 + min(a, b, c)

def word_error_rate(expected, result):
    '''
    Computes the Word Error Rate for the given two sets of words.
    '''

    return levenstein_distance(expected, result) / len(expected)
