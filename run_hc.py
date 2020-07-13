import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


if __name__ == '__main__':

    gng_output = np.load('data/gng_output.npy')

    # 階層的クラスタリング 
    hc = linkage(gng_output, metric='correlation', method='average')
    dendrogram(hc, color_threshold=0.0035)
    plt.show()

    y = fcluster(hc, 0.0035, criterion='distance')
    plt.scatter(range(len(y)), y, )
    plt.show()

    np.save('data/node_labels', y)