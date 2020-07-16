import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, centroid
from scipy.spatial.distance import pdist
import astropy.io.fits as iofits
from mylib import download_sdss_img
from astropy.wcs import WCS

def calc_coord_ave(patches):
    num = len(patches)
    x_sum, y_sum = 0, 0
    
    for p in patches:
        x_sum += p.pix_coord[0]
        y_sum += p.pix_coord[1]
    
    x = int(x_sum / num)
    y = int(y_sum / num)
    
    return x, y

if __name__ == '__main__':

    fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
    img = fits[0].data
    
    with open('data/galaxies', 'rb') as f:
        galaxies = pickle.load(f)
        
    matrix3 = np.load('data/matrix3.npy').astype(np.float64)
    
    wcs = WCS(fits[0].header)
    
    for i in range(len(matrix3)):
        matrix3[i] /= matrix3[i].sum()

    Z = centroid(pdist(matrix3, 'correlation'))
    maxclust = 6
    ct = Z[-(maxclust-1), 2]
    cluster = fcluster(Z, maxclust, criterion='maxclust')
    dendrogram(Z, color_threshold=ct)
    plt.show()
                        
    for cls in range(cluster.max()):
        cls += 1
        idxs = np.where(cluster == cls)[0]
        for idx in idxs:
            galaxy = galaxies[idx]
            if len(galaxy) == 0:
                continue
            
            x, y = calc_coord_ave(galaxy)
            # r = 20
            # plt.imshow(np.log(img[x-r:x+r, y-r:y+r]), cmap='gist_gray')
            # plt.colorbar()
            # plt.savefig('data/classificasion/' + str(cls) + '/' + str(idx))
            # plt.close()
            
            ra, dec = wcs.wcs_pix2world(y, x, 0)
            ra = ra + 0
            dec = dec + 0
            print(ra, dec)
            path = 'data/classificasion/' + str(cls) + '/' + str(idx) + '.jpg'
            download_sdss_img(path, ra, dec, 256, 256)

            
                        
    # for galaxy in galaxies:
    #     if len(galaxy) == 0:
    #         continue
        
    #     x, y = calc_coord_ave(galaxy)
    #     r = 20
    #     plt.imshow(np.log(img[x-r:x+r, y-r:y+r]), cmap='gray')
    #     plt.colorbar()
    #     plt.show()