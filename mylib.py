import numpy as np
from scipy import ndimage
from pydl.photoop.photoobj import unwrap_objid
import requests

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


def download_sdss_img(file_name, ra, dec, width, height, opt=''):
    dr7 = 'http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx'
    
    response = requests.get(
        dr7,
        params={
            'ra': ra, 
            'dec': dec, 
            'width': width, 
            'height': height,
            'opt': opt
            }
    )
    
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)


class Patch:
    def __init__(self, fits_path, pix_coord, sky_coord, img):
        self.fits_path = fits_path
        self.pix_coord = pix_coord
        self.sky_coord = sky_coord
        self.img = img
        
        self.sv = 0
        self.hc_cluster = 0
        self.galaxy = 0
        
        
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        return {r: self.members(r) for r in self.roots()}

    def __str__(self):
        return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())