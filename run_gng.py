import numpy as np
from neupy import algorithms, utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    sample_vectors = np.load('data/sample_vectors.npy')
    sample_vectors_p = np.load('data/sample_vectors_p.npy')

    sv_len = len(sample_vectors[0])

    utils.reproducible()

    gng = algorithms.GrowingNeuralGas(
        n_inputs=sv_len,
        n_start_nodes=2,

        shuffle_data=True,
        verbose=False,

        step=0.1,
        neighbour_step=0.001,

        max_edge_age=50,
        max_nodes=200,
        n_iter_before_neuron_added=200,

        after_split_error_decay_rate = 0.5,
        error_decay_rate=0.995,
        min_distance_for_update=0.2,
    )

    gng.train(sample_vectors, epochs=10)

    nodes = []
    
    for node in gng.graph.nodes:
        nodes.append(node.weight[0])

    np.save('data/gng_output', nodes)

    # inputのプロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(sample_vectors[:, 0], sample_vectors[:, 1], sample_vectors[:, 2], s=5)
    ax.set_xlabel('sample vector[0]')
    ax.set_ylabel('sample vector[1]')
    ax.set_zlabel('sample vector[2]')

    # GNGのプロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(sample_vectors[:, 0], sample_vectors[:, 1], sample_vectors[:, 2], s=5, alpha=0.3)

    for node_1, node_2 in gng.graph.edges:
        x = [node_1.weight[0][0], node_2.weight[0][0]]
        y = [node_1.weight[0][1], node_2.weight[0][1]]
        z = [node_1.weight[0][2], node_2.weight[0][2]]
        ax.scatter(x, y, z, color='orangered', zorder=2, s=10)
        ax.plot(x, y, z, color='grey', zorder=1, lw=1)

    
    ax.set_xlabel('sample vector[0]')
    ax.set_ylabel('sample vector[1]')
    ax.set_zlabel('sample vector[2]')
    plt.show()