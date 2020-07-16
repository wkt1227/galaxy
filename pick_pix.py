import astropy.io.fits as iofits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.modeling import models, fitting

path = 'data/fits/fpC-001729-r3-0083.fit.gz'
fits = iofits.open(path)
img = fits[0].data

# ガウシアンフィット
param = norm.fit(img.flatten())
fmax, fmin, nbin = 3000, 0, 300
x = np.linspace(fmin, fmax, nbin+1) + (fmax - fmin) / nbin /2
x = x[:-1]
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])
fit        = fitting.LevMarLSQFitter()
gauss_init = models.Gaussian1D(mean = x[np.argmax(pdf_fitted)], stddev = 100, amplitude = max(pdf_fitted)) #A
result     = fit(gauss_init, x, pdf_fitted)

sky = result.mean[0]
img = img - sky
img_abs = np.abs(img)

print('ガウシアンフィットのσ: {}'.format(result.stddev[0]))
print('画像の平均: {}'.format(np.mean(img_abs)))
print('画像の中央値: {}'.format(np.median(img_abs)))

# sigma = ガウシアンフィットの分散の平方根
plt.imshow(img >= 4*result.stddev[0], cmap='gray')
plt.show()

# sigma = 画像の平均
plt.imshow(img >= 4*np.mean(img_abs), cmap='gray')
plt.show()

# sigma = 画像の中央値
plt.imshow(img >= 4*np.median(img_abs), cmap='gray')
plt.show()