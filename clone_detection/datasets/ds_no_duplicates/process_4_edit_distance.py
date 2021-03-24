import pandas as pd
from os import path
import io

def save_files():
    fragments_path = 'fragments.csv'
    output_dir = './files'
    
    def save(row):
       with io.open(path.join(output_dir, '%d.java' % row['id']), 'w', encoding='utf-8') as f:
           f.write(row['code'])

    fragments_df = pd.read_csv(fragments_path)
    fragments_df.apply(lambda row: save(row), axis=1)

def filter_types():
    pairs_path = 'pairs.csv'
    output_dir = '.'

    tmp_pairs_df = pd.read_csv(pairs_path)
    pairs_dfs = list()
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 1])
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 2])
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 3])
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 4])

    for i, pairs_df in enumerate(pairs_dfs):
        i = i + 1
        pairs_df.to_csv(path.join(output_dir, 'type_%d.csv' % i), sep=',', index=False, mode='a', encoding='utf-8')

def filter_types_and_merge_codes():
    fragments_path = 'fragments.csv'
    pairs_path = 'pairs.csv'
    output_dir = '.'

    fragments_df = pd.read_csv(fragments_path)
    fragments_df = fragments_df.set_index('id')

    tmp_pairs_df = pd.read_csv(pairs_path)
    pairs_dfs = list()
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 1])
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 2])
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 3])
    pairs_dfs.append(tmp_pairs_df[tmp_pairs_df['type'] == 4])

    def merge(row):
        print(row['id1'], row['id2'])
        tmp1 = fragments_df.loc[row['id1']]
        tmp2 = fragments_df.loc[row['id2']]
        return tmp1['code'], tmp2['code']

    for i, pairs_df in enumerate(pairs_dfs):
        i = i + 1
        print('%d -- %d' % (i, pairs_df.size))
        continue
        pairs_df['code1'], pairs_df['code2'], = zip(*pairs_df.apply(lambda row: merge(row), axis=1))
        pairs_df.to_csv(path.join(output_dir, 'codes_type_%d.csv' % i), sep=',', index=False, mode='w', encoding='utf-8')

if __name__ == "__main__":
    filter_types_and_merge_codes()