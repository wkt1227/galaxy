import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as iofits
from scipy.stats import norm
from scipy.optimize import curve_fit

fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
img = fits[0].data

def gaussian_func(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

param = norm.fit(img.flatten())
print('中央値：{}, 平均値：{}'.format(np.median(img), img.mean()))
print(param)

x = np.linspace(100, 30000, 29900)
estimated_curve = gaussian_func(x, param[0], param[1])
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])
pdf = norm.pdf(x)
plt.plot(x, estimated_curve, 'r-')
plt.hist(img.flatten(), density=True, bins=100)
plt.show()
