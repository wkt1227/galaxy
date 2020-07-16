import pickle
import numpy as np

if __name__ == "__main__":
    
    with open('data/patches', 'rb') as f:
        patches = pickle.load(f)
    
    galaxy_num = 0
    p_class_num = 0
    patches = [p for p in patches if p.galaxy > 0]
    
    for patch in patches:
        galaxy_num = max(galaxy_num, patch.galaxy)
        p_class_num = max(p_class_num, patch.hc_cluster)
            
    matrix3 = np.zeros((galaxy_num, p_class_num + 1)).astype(np.float64)
    
    galaxies = [[] for _ in range(galaxy_num)]
    
    for patch in patches:
        galaxy = patch.galaxy - 1
        galaxies[galaxy].append(patch)
        p_class = patch.hc_cluster
        matrix3[galaxy, p_class] += 1 
    
    with open('data/galaxies', 'wb') as f:
        pickle.dump(galaxies, f)
        
    np.save('data/matrix3', matrix3)
    print(matrix3)