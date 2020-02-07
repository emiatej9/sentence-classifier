

from os.path import dirname, realpath, join
from keras.preprocessing.text import Tokenizer

def test_a():
    root_dir = dirname(dirname(realpath(__file__)))
    path_txt = join(root_dir, 'data/nsmc/test/sentences.txt')
    num_words = 345788


    t = Tokenizer(num_words=num_words)

    with open(path_txt) as f:
        lines = f.readlines()
        t.fit_on_texts(lines)



    print(t.word_index)
