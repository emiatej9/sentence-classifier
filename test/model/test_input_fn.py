
from os.path import dirname, realpath, join
from model.input_fn import load_dataset_from_text, index_table_from_text


def test_index_table_from_text():
    root_dir = dirname(dirname(dirname(realpath(__file__))))
    path_txt = join(root_dir, 'data/small/train/source.txt')

    symbol_index = index_table_from_text(path_txt)
    assert '<sos>' not in symbol_index
    assert '<eos>' not in symbol_index

    symbol_index = index_table_from_text(path_txt, sos=True, eos=True)
    assert '<sos>' in symbol_index
    assert '<eos>' in symbol_index
    print(symbol_index)


def test_load_dataset_from_text():
    root_dir = dirname(dirname(dirname(realpath(__file__))))
    path_txt = join(root_dir, 'data/small/train/source.txt')

    symbol_index = index_table_from_text(path_txt, sos=True, eos=True)
    dataset = load_dataset_from_text(join(root_dir, path_txt), symbol_index)

    with open(join(root_dir, path_txt), 'r') as f:
        num_lines = len(f.read().strip().split('\n'))
        assert num_lines == len(dataset)

    dataset_with_mark = load_dataset_from_text(join(root_dir, path_txt), symbol_index, sos=True, eos=True)
    assert len(dataset[0]) + 2 == len(dataset_with_mark[0])

