import numpy as np
from scipy import ndimage
from pydl.photoop.photoobj import unwrap_objid


# 画像からパワースペクトルを計算する
def get_psd2d(img):
    # 2次元FFT
    img_fft = np.fft.fft2(img)

    # 象限の入れ替え
    img_fft = np.fft.fftshift(img_fft)

    # パワースペクトル
    img_power = 20*np.log(np.abs(img_fft))

    return img_power


# パワースペクトルの半径方向の平均を計算する
def get_psd1d(psd2d):

    h = psd2d.shape[0]
    w = psd2d.shape[1]
    wc = w // 2
    hc = h // 2

    # psd2dの中心からの距離の配列を作る
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc).astype(np.int)

    # rを用いて、psd2dの半径方向の平均を計算する
    psd1d = ndimage.mean(psd2d, r, index = np.arange(0, wc + 1))

    return psd1d


# 銀河のobjidから、その銀河が含まれるfitsファイルの名前を得る
def get_fits_name_from_objid(objid, f = 'r'):

    params = unwrap_objid(objid)
    run = str(params['run'])
    camcol = str(params['camcol'])
    frame = str(params['frame'])

    fits_name = 'fpC-' + run.zfill(6) + '-' + f + camcol + '-' + frame.zfill(4) + 'fit.gz'

    return fits_name

class Patch:
    def __init__(self, center_coord, img, sample_vector):
        self.coord = center_coord
        self.img = img
        self.sv = sample_vector