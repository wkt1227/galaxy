import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#===================================================================
# Get PSD 1D (mean radial power spectrum)
#===================================================================
def GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc+1))

    return psd1D

def calc_powerspectrum(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_power = 20*np.log(np.abs(img_fft))
    
    return img_power

if __name__ == '__main__':

    # 画像読込
    npz = np.load('data/npz/587727178986356823.npz')
    img = npz['arr_0'][0:8, 0:8]

    # 2次元FFT
    img_fft = np.fft.fft2(img)

    # 象限の入れ替え
    img_fft = np.fft.fftshift(img_fft)

    # パワースペクトル
    img_power = 20*np.log(np.abs(img_fft))
    print(img_power)

    print(GetPSD1D(img_power))

    # 画像として保存
    plt.imshow(img_power, cmap = 'gray')
    plt.savefig('587727178986356823_p', bbox_inche = 'tight')
    plt.close()