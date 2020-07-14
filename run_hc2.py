import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import astropy.io.fits as iofits

def calc_coord_ave(patches):
    num = len(patches)
    x, y = 0, 0
    
    for p in patches:
        x += p.coord[0]
        y += p.coord[1]
    
    x = int(x / num)
    y = int(y / num)
    
    return x, y

if __name__ == '__main__':

    fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
    img = fits[0].data
    f = open('data/galaxies', 'rb')
    galaxies = pickle.load(f)
    matrix3 = np.load('data/matrix3.npy').astype(np.float32)
    for i in range(len(matrix3)):
        if matrix3[i].sum() == 0:
            continue
         
        matrix3[i] /= matrix3[i].sum()
    print(matrix3)

    hc = linkage(matrix3, metric='euclidean', method='average')
    dendrogram(hc, color_threshold=0.15, orientation='right')
    plt.show()
    
    cluster = fcluster(hc, 0.15, criterion='distance')
                        
    for cls in range(cluster.max()):
        cls += 1
        idxs = np.where(cluster == cls)[0]
        for idx in idxs:
            galaxy = galaxies[idx]
            if len(galaxy) == 0:
                continue
            
            x, y = calc_coord_ave(galaxy)
            r = 20
            plt.imshow(np.log(img[x-r:x+r, y-r:y+r]), cmap='gray')
            plt.colorbar()
            plt.savefig('data/classificasion/' + str(cls) + '/' + str(idx))
            plt.close()
            
                        
    # for galaxy in galaxies:
    #     if len(galaxy) == 0:
    #         continue
        
    #     x, y = calc_coord_ave(galaxy)
    #     r = 20
    #     plt.imshow(np.log(img[x-r:x+r, y-r:y+r]), cmap='gray')
    #     plt.colorbar()
    #     plt.show()