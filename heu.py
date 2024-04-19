import nltk
from nltk.corpus import words
from nltk.corpus import brown
from collections import Counter

# Ensure that the 'words' corpus is downloaded
nltk.download('words')
nltk.download('brown')

word_freq = nltk.FreqDist(w.lower() for w in brown.words())
common_words = {word for word, freq in word_freq.items() if freq > 3}

def decode_sequence(input_string, min_tr=4, max_tr=12, max_mistake=3, model='brown', max_return=1):
    """
    we expect input_string with null char as "@", which is produced by
    cnn with low logit. In this case we will not consider the frame as
    a valid character, like changing pose.
    using "%" represent EOS
    :param max_return: max_return number of possible answer and certainty
    :param max_mistake: max mistake char that can appear on the continue frame
    :param input_string: a list of chars
    :param min_tr: min continue frames to consider a valid char
    :param max_tr: max continue frames to consider a single valid char
    :param model: which dictionary to use
    :return: max_return number of possible answer and certainty
    """
    # English words list
    if model == 'brown':
        word_list = common_words
    elif model == 'word':
        word_list = words.words()

    guess = ''
    current_char = None
    count = 0
    mistake = 0

    # in test case the string may not include a '%' as EOS
    if not input_string.endswith("%"):
        input_string += "%"

    for c in input_string:
        # init state
        if c == "@":
            continue
        if current_char is None:
            current_char = c
            count += 1
            continue
        if c == current_char and c != "%":
            count += 1
            mistake = 0
            continue
        elif count > min_tr and c != current_char and mistake < max_mistake and c != "%":
            mistake += 1
            if mistake == max_mistake:
                pass
            else:
                continue

        # evaluate the word
        if count < min_tr:
            # not consider as valid char
            current_char = c
            count = 1
        elif count > max_tr:
            # should consider as multiple char
            times = count // max_tr + 1
            for _ in range(0, times):
                guess += current_char
            current_char = c
            count = 1
        else:
            # should consider as single char
            guess += current_char
            current_char = c
            count = 1

        mistake = 0

    # return guess

    potential_words = {guess: 10}

    # generate one char missing
    for index in range(0, len(guess)):
        c = guess[index]
        w = guess[:index] + c + guess[index:]
        potential_words[w] = 2
    # generate one char wrong
    # generate one char more
    for index in range(1, len(guess)):
        c = guess[index]
        w = guess[:index-1] + guess[index:]
        potential_words[w] = 1



    # Filter by checking against the English words list
    potential_words_key = potential_words.keys()
    valid_words = [word for word in potential_words_key if word in word_list]

    # Find the most probable word
    if valid_words:
        valid_words = sorted(valid_words, key=lambda x: potential_words[x])
        temp = valid_words[:max_return]
        return temp
    else:
        return "No valid word found"


# Test the function
# input_string = "hhhhhhhabdyuehbnioufbacviubdfvueeeeeeebdyuehbnioufballlllllllbdyuehbnioufbaooooobdyuehbniobdyuehbni"
# input_string = 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGXCCOQOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOQOQQQTTTTTTTTTTTTTDVTVVVDVVVVVVVDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD'
# input_string = input_string.lower()
# decoded_word = decode_sequence(input_string, max_tr=240, min_tr=30, max_return=2)
# print(decoded_word)
