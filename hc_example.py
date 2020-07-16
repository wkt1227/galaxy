import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, centroid
from scipy.spatial.distance import pdist


if __name__ == "__main__":
    
    gng_output = np.load('data/gng_output.npy')
    
    # 階層的クラスタリング(距離: 相関係数, 手法: 重心法)
    Z = centroid(pdist(gng_output, 'correlation'))
    
    # 距離を閾値としてクラスタリングする
    cl_dist = fcluster(Z, 0.0035, criterion='distance')
    dendrogram(Z, color_threshold=0.0035)
    plt.axhline(0.0035, linestyle='--', c='purple')
    plt.show()
    
    # 最大クラスタ数を決めてクラスタリングする
    maxclust = 15
    ct = Z[-(maxclust-1), 2]
    cl_maxclust = fcluster(Z, maxclust, criterion='maxclust')
    dendrogram(Z, color_threshold=ct)
    plt.axhline(ct, linestyle='--', c='purple')
    plt.show()
    print(cl_maxclust)