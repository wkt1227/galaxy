import astropy.io.fits as iofits
import numpy as np
import matplotlib.pyplot as plt
import mylib
import statistics
from mpl_toolkits.axes_grid1 import make_axes_locatable


fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
img = fits[0].data
img_p = img - img.min()

patches = []
sample_vectors = []
p_sample_vectors = []
coordinates = np.array(np.where(img_p >= img_p.mean()*4)).T

p_size = 6
for x, y in coordinates:
    patch = img[x-p_size:x+p_size, y-p_size:y+p_size]
    patches.append(patch)

for i, patch in enumerate(patches[:100]):
    # パワースペクトル
    power_spectrum = mylib.get_psd2d(patch)
    # sample vector(パワースペクトルの半径方向の平均)
    sample_vector = mylib.get_psd1d(power_spectrum)
    sample_vectors.append(sample_vector)
    # sample vectorの加工(対数分布→正規分布、正規化)
    p_sample_vector = np.log(sample_vector)
    p_sample_vector = p_sample_vector - p_sample_vector.mean()
    p_sample_vector = p_sample_vector / statistics.pstdev(p_sample_vector)

    fig = plt.figure(figsize=(9, 3))

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(patch, cmap='gray')
    ax1.set_title('patch')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(power_spectrum)
    ax2.set_title('power spectrum')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(range(len(sample_vector)), sample_vector, '-s')
    ax3.set_title('sample vector(radially average)')

    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.plot(range(len(p_sample_vector)), p_sample_vector, '-s')
    # ax4.set_title('processed sample vector')

    fig.tight_layout()
    fig.show()
    fig.savefig('data/fig2/fig{}.png'.format(str(i)))
    plt.close(fig)