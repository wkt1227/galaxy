import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    output_path = 'data/gng_output.npy'
    input_path = 'data/sample_vectors.npy'

    gng_output = np.load(output_path)
    gng_input = np.load(input_path)
    input_labels = []

    # クラスタリング
    for i, v in enumerate(gng_input):

        nearest_idx = 0
        dist = 0

        for j, u in enumerate(gng_output):
            # ユークリッド距離の計算
            tmp_dist = np.linalg.norm(v-u)
            if j == 0:
                dist = tmp_dist
                continue

            if tmp_dist < dist:
                dist = tmp_dist
                nearest_idx = j
        
        input_labels.append(nearest_idx)
    
    # プロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(gng_output[:, 0], gng_output[:, 1], gng_output[:, 2], c=np.arange(len(gng_output)), s=50, zorder=1, cmap=cm.hsv, alpha=0.6)
    sc = ax.scatter(gng_input[:, 0], gng_input[:, 1], gng_input[:, 2], c=np.array(input_labels), s=2, zorder=2, cmap=cm.hsv)
    plt.colorbar(sc)
    plt.show()
