import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as iofits
from scipy.stats import norm
from astropy.modeling import models, fitting #A

# Params
fmax, fmin, nbin = 3000, 0, 300 #A

fits = iofits.open('data/fits/fpC-001729-r3-0083.fit.gz')
img = fits[0].data
param = norm.fit(img.flatten())
print(param[0], param[1])

print('中央値：{}, 平均値：{}'.format(np.median(img), img.mean()))
x = np.linspace(fmin, fmax, nbin+1) + (fmax - fmin) / nbin /2  #A
x = x[:-1]#A
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])  #A

fit        = fitting.LevMarLSQFitter() #A
gauss_init = models.Gaussian1D(mean = x[np.argmax(pdf_fitted)], stddev = 100, amplitude = max(pdf_fitted)) #A
result     = fit(gauss_init, x, pdf_fitted) #A
print(result, result.mean[0])

print(gauss_init)

plt.plot(x, pdf_fitted, 'r-')
plt.plot(x, result(x), 'b.')
plt.xlim([0, fmax]) #A
plt.show()
