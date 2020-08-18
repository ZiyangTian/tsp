import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tsp(prefix):
    parameters = np.array(pd.read_csv(prefix + '.parameters', header=None))
    rank = np.array(pd.read_csv(prefix + '.rank', header=None))[0].tolist()
    rank = np.array(rank + rank[0:1])

    plt.figure()
    plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
    plt.plot(parameters[:, 0][rank], parameters[:, 1][rank], 'r-', color='blue')
    plt.show()


plot_tsp(r'E:\Programs\DataSets\tsp\84')