import os
import glob
import numpy as np
import pandas as pd
import random


def main():
    pattern = r'E:\Programs\DataSets\tsp\tsp-4\*'
    files = glob.glob(pattern)
    bases = list(set(map(lambda f: f.split('.')[0], files)))
    # print(bases)
    random.shuffle(bases)

    data = []
    for b in bases:
        parameters = np.array(pd.read_csv(b + '.parameters', header=None)).reshape((-1,)).tolist()
        rank = np.array(pd.read_csv(b + '.rank', header=None)).squeeze()
        for i in range(len(rank)):
            if rank[i] == 0:
                rank = list(rank[i:]) + list(rank[:i])
                break
        data.append((parameters, rank))
    with open(r'E:\Programs\DataSets\tsp\data4.txt', 'w') as f:
        for d in data:
            f.write(','.join(map(str, d[0])) + ';' + ','.join(map(str, d[1])) + '\n')


if __name__ == '__main__':
    main()
