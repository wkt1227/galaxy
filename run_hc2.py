from ccl import galaxy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pickle


if __name__ == '__main__':

    sv_labels = np.load('data/input_labels.npy') - 1
    f = open('data/galaxies.txt', 'rb')
    galaxies = pickle.load(f)
    matrix3 = np.zeros_like((len(galaxies), 24))
    
    for galaxy in galaxies:

        for x, y in galaxy:

            