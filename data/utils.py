import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import datasets


def plot_tsp(prefix):
    parameters = np.array(pd.read_csv(prefix + '.parameters', header=None))
    rank = np.array(pd.read_csv(prefix + '.rank', header=None))[0].tolist()
    rank = np.array(rank + rank[0:1])

    plt.figure()
    plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
    plt.plot(parameters[:, 0][rank], parameters[:, 1][rank], 'r-', color='blue')
    plt.show()


def main():
    pattern = r'../demo/data.txt'
    dl = datasets.TSPDataLoader(pattern, batch_size=1)
    for i, (p, r, _) in enumerate(dl):
        if i == 3:
            parameters = p[:, 0, :].numpy()
            rank = r[:, 0].numpy()
            rank = np.concatenate([rank, rank[0:1]], axis=0)
            print(rank)

            plt.figure()
            plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
            plt.plot(parameters[:, 0][rank], parameters[:, 1][rank], 'r-', color='blue')
            plt.show()

            break


if __name__ == '__main__':
    main()
