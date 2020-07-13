import astropy.io.fits as iofits
import numpy as np
import mylib
import statistics


if __name__ == '__main__':

    fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
    img = fits[0].data
    img_p = img - img.min()
    sigma = img_p.mean()

    patches = []
    sample_vectors = []
    sample_vectors_p = []
    coordinates = np.array(np.where(img_p >= 4*sigma)).T

    patch_size = 6

    for x, y in coordinates:
        patch = img_p[x-patch_size:x+patch_size, y-patch_size:y+patch_size]
        patches.append(patch)

    for patch in patches:
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

    np.save('data/sample_vectors', sample_vectors)
    np.save('data/sample_vectors_p', sample_vectors_p)