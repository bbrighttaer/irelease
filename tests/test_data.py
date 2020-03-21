import unittest
from gpmt.data import GeneratorData

gen_data_path = '../data/chembl_small.smi'
tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3',
          '2', '5', '4', '7', '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I',
          'H', 'O', 'N', 'P', 'S', '[', ']', '\\', 'c', 'e', 'i', 'l', 'o', 'n',
          'p', 's', 'r', '\n']
print(f'Number of tokens = {len(tokens)}')
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[1], keep_header=False, tokens=tokens)


class MyTestCase(unittest.TestCase):

    def test_batch(self):
        bz = 32
        batch = gen_data.random_training_set(batch_size=bz)
        assert(len(batch[0]) == bz and len(batch[1]) == bz)


if __name__ == '__main__':
    unittest.main()
