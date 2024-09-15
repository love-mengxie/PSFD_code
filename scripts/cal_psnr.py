from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from PIL import Image

def evaluation(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    # print('MSE =', mse(img1, img2))
    # print('PSNR =', psnr(img1, img2))
    # print('SSIM =', ssim(img1, img2, channel_axis=2))
    return mse(img1, img2), psnr(img1, img2), ssim(img1, img2, channel_axis=2)

def main():
    res_ori_folder = '/root/autodl-tmp/results_ori'
    res_adv_folder = '/root/autodl-tmp/results005'
    file_names = os.listdir(res_adv_folder)
    num = len(file_names)
    res = 0
    res_mse = 0
    flag = 0
    res_ssim = 0
    for file_name in file_names:
    # file_name = '000000688350.png'
        # if_name = '000000688350'
        adv_path = os.path.join(res_adv_folder, file_name)
        name = os.path.basename(adv_path)[:12]
        if name[0] != '0':
             continue
        # print(adv_path)
        ori_path = os.path.join(res_ori_folder, f"{name}.png")
        adv = Image.open(adv_path).convert("RGB")
        ori = Image.open(ori_path).convert("RGB")
        mse, psnr, ssim = evaluation(adv, ori)
        if psnr == float('inf') or mse == float('inf'):
            continue
        # print(psnr)
        res += psnr
        res_mse += mse
        res_ssim += ssim
        flag += 1
        
    print(num)
    print(flag)
    
    print(res/flag)
    print(res_mse/flag)
    print(res_ssim/flag)
    
if __name__ == "__main__":
    main()