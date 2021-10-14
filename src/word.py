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
