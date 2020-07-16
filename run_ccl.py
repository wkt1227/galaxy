import astropy.io.fits as iofits
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
import pickle

path = 'data/fits/fpC-001729-r3-0083.fit.gz'

fits = iofits.open(path)
img = fits[0].data

with open('data/patches', 'rb') as f:
    patches = pickle.load(f)

patch_ids = [(p.fits_path, p.pix_coord) for p in patches]

# ガウシアンフィット
mean, std = norm.fit(img.flatten())

img_p = img - mean
img_p_abs = np.abs(img_p)
img_pp = np.zeros_like(img_p)
# sigma = np.mean(img_p_abs)
sigma = std
picked_pix_coords = np.array(np.where(img_p >= 4*sigma)).T
p_size = 6

for x, y in picked_pix_coords:
    patch = img[x-p_size:x+p_size, y-p_size:y+p_size]
    mean = np.mean(patch)
    img_pp[x, y] = mean

img_b = cv2.threshold(img_pp, 4*sigma, 255, cv2.THRESH_BINARY)[1].astype('uint8')

plt.imshow(img_pp, cmap='gray')
plt.colorbar()
plt.show()

n_labels, label_imgs = cv2.connectedComponents(img_b)
plt.imshow(label_imgs)
plt.colorbar()
plt.show()

for i in range(1, n_labels):
    galaxy = np.array(np.where(label_imgs == i)).T
    # if len(galaxy) <= 5:
    #     continue
    
    for x, y in galaxy:
        patch_id = patch_ids.index((path, (x, y)))
        patches[patch_id].galaxy = i
        
with open('data/patches', 'wb') as f:
    pickle.dump(patches, f)

