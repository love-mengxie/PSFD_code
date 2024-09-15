from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import lpips
from PIL import Image

class UtilOfLpips():
    def __init__(self, net, use_gpu=False):
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img0, img1):
        # Load images
        img0 = lpips.im2tensor(img0)  # RGB image from [-1,1]
        img1 = lpips.im2tensor(img1)

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01
        
def evaluation(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    lpips_util = UtilOfLpips(net='alex')
    return mse(img1, img2), psnr(img1, img2), ssim(img1, img2, channel_axis=2), lpips_util.calc_lpips(img1, img2)


#img_adv =Image.open("1.png").convert("RGB")
#tar_adv=Image.open("2.png").convert("RGB")    
#mse,psnr,ssim,lpips=evaluation(img_adv,tar_adv)
##print("psnr=",psnr)
