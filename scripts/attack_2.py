import os
import random

from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T

from utils import preprocess, recover_image

def pgd(X, targets, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape) * 2 * eps - eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - targets).norm()

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])

        # X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = X_adv + grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if mask is not None:
            X_adv.data *= mask

    return X_adv

def get_target():
    # target_url = 'https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png'
    # target_url = 'https://img95.699pic.com/photo/50010/6735.jpg_wh860.jpg'
    # response = requests.get(target_url)
    # target_image = Image.open(BytesIO(response.content)).convert("RGB")
    # target_image = target_image.resize((512, 512))
    # target_image = Image.open('./examples/image/example_3.png').convert("RGB")
    # target_image = target_image.resize((512, 512))
    target_image = Image.new("RGB", (512, 512), color="black")

    return target_image


def main():
    print("start!")
    to_pil = T.ToPILImage()

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "/root/autodl-tmp/diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe_img2img = pipe_img2img.to("cuda")

    image_folder = "/root/autodl-tmp/test_bench/GT_3500"
    attack_folder = "/root/autodl-tmp/max_attack011"
    image_files = os.listdir(image_folder)
    adv_image_files = os.listdir(attack_folder)

    image_files = random.sample(image_files, 700)
    f = 0
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        filename = os.path.basename(image_path)[:12]
        # if any(filename == adv_file[:12] for adv_file in adv_image_files):
        #      continue
        if filename[0] != '0':
            continue
        f += 1
        print(f)
        adv_path = os.path.join("/root/autodl-tmp/max_attack011", f"{filename}.png")
        # image_path = "./autodl/examples/image/example_1.png"
        init_image = Image.open(image_path).convert("RGB")
        resize = T.Resize(512)
        center_crop = T.CenterCrop(512)
        init_image = center_crop(resize(init_image))

        # 将图像转换为 numpy 数组，以便 prepare_mask_and_masked_image 函数处理
        init_image = np.array(init_image)

        X = init_image[None].transpose(0, 3, 1, 2)
        X = torch.from_numpy(X).to(dtype=torch.float32) / 127.5 - 1.0

        # attack
        with torch.autocast('cuda'):
            X = X.half().cuda()
            # targets = pipe_img2img.vae.encode(preprocess(get_target()).half().cuda()).latent_dist.mean
            targets = pipe_img2img.vae.encode(X).latent_dist.mean
            adv_X = pgd(X,
                        targets=targets,
                        model=pipe_img2img.vae.encode,
                        clamp_min=-1,
                        clamp_max=1,
                        eps=0.11,
                        step_size=0.01,
                        iters=300,
                        )

            # 将像素值转换回 [0,1] 范围
            adv_X = (adv_X / 2 + 0.5).clamp(0, 1)

            adv_image = T.ToPILImage()(adv_X[0]).convert("RGB")
            adv_image.save(adv_path)
        # adv_image.save(adv_path)

if __name__ == '__main__':
    main()

# 随机选取700张图像