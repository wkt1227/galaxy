import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as iofits
from neupy import algorithms, utils
import mylib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram
import statistics

fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
img = fits[0].data
img_p = img - img.min()

print('[加工前] min:{}, max:{}, mean:{}'.format(img.min(), img.max(), img.mean()))
print('[加工後] min:{}, max:{}, mean:{}'.format(img_p.min(), img_p.max(), img_p.mean()))

# plt.imshow(img_p[397-50:397+50, 157-50:157+50] >= img_p.mean()*4)
# plt.imshow(img_p >= img_p.mean()*4)
# plt.colorbar()
# plt.show()

# print((img_p[397-50:397+50, 157-50:157+50] >= img_p.mean()*4).sum())

patches = []
sample_vectors = []
p_sample_vectors = []
coordinates = np.array(np.where(img_p >= img_p.mean()*4)).T

p_size = 6
for x, y in coordinates:
    patch = img[x-p_size:x+p_size, y-p_size:y+p_size]
    patches.append(patch)

for patch in patches:
    # パワースペクトル
    power_spectrum = mylib.get_psd2d(patch)
    # sample vector(パワースペクトルの半径方向の平均)
    sample_vector = mylib.get_psd1d(power_spectrum)
    sample_vectors.append(sample_vector)
    # sample vectorの加工(対数分布→正規分布、正規化)
    p_sample_vector = np.log(sample_vector)
    p_sample_vector = p_sample_vector - p_sample_vector.mean()
    p_sample_vector = p_sample_vector / statistics.pstdev(p_sample_vector)
    p_sample_vectors.append(p_sample_vector)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(sample_vectors[:][0], sample_vectors[:][2], sample_vectors[:][6])
# plt.scatter(sample_vectors[:][0], sample_vectors[:][6], s=5)
# plt.show()

gng = algorithms.GrowingNeuralGas(
    n_inputs=p_size+1,
    n_start_nodes=2,

    shuffle_data=True,
    verbose=False,

    step=0.2,
    neighbour_step=0.001,

    max_edge_age=30,
    max_nodes=500,

    n_iter_before_neuron_added=100,
    after_split_error_decay_rate=0.5,
    error_decay_rate=0.995,
    min_distance_for_update=0.01,
)

# plt.scatter(sample_vectors[:][0], sample_vectors[:][6], alpha=0.5, s=5)

gng.train(sample_vectors, epochs=10)

for node_1, node_2 in gng.graph.edges:
    x = [node_1.weight[0][0], node_2.weight[0][0]]
    y = [node_1.weight[0][2], node_2.weight[0][2]]
    z = [node_1.weight[0][6], node_2.weight[0][6]]
    ax.plot(x, y, z, color='black')
    ax.set_xlabel('sample vector[0]')
    ax.set_ylabel('sample vector[2]')
    ax.set_zlabel('sample vector[6]')

nodes = []
for node in gng.graph.nodes:
    nodes.append(node.weight[0])
#     plt.scatter(node.weight[0][0], node.weight[0][6], color='red')

plt.show()

# hc = linkage(nodes, metric='correlation', method='single') 
# dendrogram(hc)
# plt.show()