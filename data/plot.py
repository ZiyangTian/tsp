import numpy as np
import matplotlib.pyplot as plt


def plot_tsp(parameters, rank):
    rank = np.concatenate([rank, rank[0:1]], axis=0)

    plt.figure()
    plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
    plt.plot(parameters[:, 0][rank], parameters[:, 1][rank], 'r-', color='blue')
    plt.show()
