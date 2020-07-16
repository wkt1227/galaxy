import requests
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageOps

    
if __name__ == "__main__":
    
    file_name = 'img.jpg'
    dr12 = 'http://skyserver.sdss.org/dr12/SkyserverWS/ImgCutout/getjpeg'
    dr7  = 'http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx'
    
    response = requests.get(
        dr7,
        params={'ra': 1.71492E-3, 
                'dec': -10.37380123, 
                'width': 256, 
                'height': 256, 
                'scale': 0.2, 
                'opt': 'GL'}
    )
    
    if response.status_code == 200:  # 200 = アクセス成功
        
        # 保存
        # with open(file_name, 'wb') as f:
        #     f.write(response.content)
        
        # 表示のみ
        img = Image.open(BytesIO(response.content))
        img.show()