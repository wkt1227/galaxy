import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, centroid
from scipy.spatial.distance import pdist

if __name__ == '__main__':

    gng_output = np.load('data/gng_output.npy')

    # 階層的クラスタリング
    Z = centroid(pdist(gng_output, 'correlation'))
    dendrogram(Z, color_threshold=0.0035)
    plt.show()
    
    # 階層的クラスタリング 
    # Z = linkage(gng_output, metric='correlation', method='average')
    # dendrogram(Z, color_threshold=0.0035)
    # plt.show()

    cl = fcluster(Z, 0.0035, criterion='distance')

    np.save('data/node_labels', cl)