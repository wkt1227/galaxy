import numpy as np
import pickle
import matplotlib.pyplot as plt


def ccl(patches):
    
    els = [[] for _ in range(len(patches))]
    p_crds = np.array([p.pix_coord for p in patches])
    p_size = int(patches[0].img.shape[0] / 2)
    label_cursor = 0
    
    for i, p in enumerate(patches):
        x, y = p.pix_coord
        nbr_p_crds = []
        n = np.where((p_crds <= np.array([x+p_size, y+p_size])) & (p_crds >= np.array([x-p_size, y-p_size])))
        nbr_label = -1
    
        for i in range(0, len(n[0]), 2):
            idx = n[0][i]
            if patches[idx].galaxy == -1:
                continue
            nbr_label = min(nbr_label, patches[idx].galaxy)
            # plt.scatter(p_crds[idx][0], p_crds[idx][1], c='r')
            
        # plt.show()
        patches[i].galaxy = nbr_label
        
                                
    
if __name__ == "__main__":
    
    with open('data/patches', 'rb') as f:
        patches = pickle.load(f)
        
    ccl(patches)