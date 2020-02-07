"""Read, split and save the nsmc dataset for our model"""

import os
import re
import json
import MeCab

from collections import Counter


def load_dataset(dir_path: str, except_first=False):
    if not os.path.exists(dir_path):
        msg = "{} doesn't exist.".format(dir_path)
        raise Exception(msg)

    dataset = list()
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        start = int(except_first)
        for line in lines[start:]:
            dataset.append(tuple(line.strip().split('\t')))

    return dataset


def preprocess(dataset: list) -> list:
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
    preprocessed = list()
    for nid, raw_sentence, raw_label in dataset:

        try:
            sentence = _preprocess(raw_sentence, tagger)
            label = int(raw_label)

            assert sentence
            assert label in (0, 1)

            preprocessed.append((sentence, label))

        except (ValueError, AssertionError):
            pass
            # line = line.strip()
            # print(f"fail to parse '{line}'.")

    return preprocessed


def _preprocess(sentence: str, tagger) -> str:

    substitutions = [('ㅡㅡ', '짜증'), ('ㅋㅋ', '웃음'), ('ㅎㅎ', '웃음'), ('노잼', '재미업다')]
    for sub in substitutions:
        sentence = sentence.replace(sub[0], f' {sub[1]} ')

    trimmed = re.sub(r'[^가-힣\d\w]', ' ', sentence).strip()

    terms = list()
    tag_symbols = list()
    if trimmed:
        morphemes = tagger.parse(trimmed).split('\n')
        end_term = 'EOS'
        ignored_tag = ('E', 'JK', 'XP', 'XS', 'I', 'NP', 'SN', 'NNBC', 'SL', 'NR', 'UNKNOWN', 'VCP')

        for morp in morphemes:
            term_tag = morp.split()
            term = term_tag[0]
            if term == end_term:
                break

            tag = term_tag[1]
            tag_symbol = tag.split(',')[0]
            if tag_symbol.startswith(ignored_tag):
                continue

            elif tag_symbol == 'NNP':
                nnp_category = tag.split(',')[1]

                if nnp_category != '*':
                    term = nnp_category

            elif tag_symbol in ('VV', 'VA', 'VX'):
                if len(term) < 2:
                    term += '다'

            if len(term) < 2:
                continue
            elif len(term) > 3:
                term = term[:2]

            terms.append(term)
            tag_symbols.append(tag_symbol)

    '''
    print(sentence)
    print(trimmed)
    for i in range(len(terms)):
        print(f'{terms[i]}({tag_symbols[i]})', end=' ')

    print('\n')
    '''
    return ' '.join(terms)


def save_dataset(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as f:
        for sentence, _ in dataset:
            f.write(f'{sentence}\n')

    with open(os.path.join(save_dir, 'labels.txt'), 'w') as f:
        for _, label in dataset:
            f.write(f'{label}\n')


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def vocabulary(sentences: list) -> Counter:
    word_counter = Counter()
    for sentence in sentences:
        splits = sentence.strip().split()
        # valid_splits = [s for s in splits if len(s) > 0]
        word_counter.update(splits)
    return word_counter


def max_length(sentences: list) -> int:
    max_len = 0
    for sentence in sentences:
        max_len = max(len(sentence.strip().split()), max_len)
    return max_len


if __name__ == "__main__":
    _dataset = preprocess(load_dataset('./data/nsmc/raw', except_first=True))
    _sentences = [sentence for sentence, label in _dataset]
    vocab = vocabulary(_sentences)

    import matplotlib.pyplot as plt

    words = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)[:100]
    print(words)
    counts = [vocab[word] for word in words]

    plt.plot(words, counts)
    plt.show()

    max_sentence_length = max_length(_sentences)
    dataset_size = len(_sentences)

    train_dataset = _dataset[:int(0.7 * dataset_size)]
    dev_dataset = _dataset[int(0.7 * dataset_size): int(0.85 * dataset_size)]
    test_dataset = _dataset[int(0.85 * dataset_size):]

    save_dataset(train_dataset, './data/nsmc/train')
    save_dataset(dev_dataset, './data/nsmc/dev')
    save_dataset(test_dataset, './data/nsmc/test')

    params = {
        'train_size': len(train_dataset),
        'dev_size': len(dev_dataset),
        'test_size': len(test_dataset),
        'vocab_size': len(vocab),
        'max_sentence_length': max_sentence_length,
    }

    with open('params/dataset_params.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)
