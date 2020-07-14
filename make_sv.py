import astropy.io.fits as iofits
import numpy as np
import mylib
import statistics
from mylib import Patch
import pickle
from scipy.stats import norm


if __name__ == '__main__':

    path = 'data/fits/fpC-001729-r3-0083.fit.gz'

    fits = iofits.open(path)
    img = np.array(fits[0].data).astype(np.float64)
    
    # ガウシアンフィット
    peak, sigma = norm.fit(img.flatten())
    img -= peak

    patches = []
    patches2 = []
    sample_vectors = []
    sample_vectors_p = []
    coordinates = np.array(np.where(img >= 4*sigma)).T

    patch_size = 6

    for x, y in coordinates:
        patch = img[x-patch_size:x+patch_size, y-patch_size:y+patch_size]
        patches.append(patch)
        patches2.append(Patch(path, (x, y), patch))

    for i, patch in enumerate(patches):
        # パワースペクトル
        power_spectrum = mylib.get_psd2d(patch)
        # sample vector(パワースペクトルの半径方向の平均)
        sample_vector = mylib.get_psd1d(power_spectrum)
        sample_vectors.append(sample_vector)
        # sample vectorの加工(対数分布→正規分布、正規化)
        sample_vector_p = np.log(sample_vector)
        sample_vector_p = sample_vector_p - sample_vector_p.mean()
        sample_vector_p = sample_vector_p / statistics.pstdev(sample_vector_p)
        sample_vectors_p.append(sample_vector_p)
        
        patches2[i].sv = sample_vector_p

    np.save('data/sample_vectors', sample_vectors)
    np.save('data/sample_vectors_p', sample_vectors_p)
    f = open('data/patches', 'wb')
    pickle.dump(patches2, f)