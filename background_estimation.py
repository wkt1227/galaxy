import astropy.io.fits
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt
from photutils import Background2D, MedianBackground
import numpy as np

def calc_background_rms(fits):
    image = fits[0].data

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (50, 50), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    rms = np.sqrt((bkg.background**2).mean())

    return rms

fits = astropy.io.fits.open('data/fits/fpC-001729-r3-0083.fit.gz')
image = fits[0].data

r = 50
row = 397
col = 157
plt.imshow(image[row-r:row+r, col-r:col+r])
plt.colorbar()
plt.show()

plt.imshow(image)
plt.colorbar()
plt.show()

sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
bkg = Background2D(image, (50,50), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
# plt.imshow(bkg.background)
# plt.colorbar()
# plt.show()
rms = np.sqrt((bkg.background**2).mean())
print(bkg.background.mean(), rms)

image2 = np.zeros_like(image)
image2 = (image >= 5*rms)
plt.imshow(image2)
plt.colorbar()
plt.show()