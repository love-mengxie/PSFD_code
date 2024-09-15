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
import torch.nn as nn
import torch.optim as optim
from DWT import DWT_2D_tiny, IDWT_2D_tiny
from torch.autograd import Variable
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_adversarial_sample(image_path: str,
                                targets,
                                model,
                                num_iteration: int = 300,
                                learning_rate: float = 0.01,
                                lambda_sa: float = 0.9,
                                lambda_fa: float = 0.1,
                                lambda_lf: float = 1,
                                lambda_hf: float = 0.01,
                                epsilon: float = 0.04,
                                wave: str = 'haar') -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    inputs = transform(image).unsqueeze(0).to(device)
    if inputs.shape[1] == 1:
        return None, None
    DWT = DWT_2D_tiny(wavename=wave)
    IDWT = IDWT_2D_tiny(wavename=wave)

    inputs_ll = DWT(inputs)
    inputs_hf = inputs - IDWT(inputs_ll)

    eps = 3e-9
    modifier = torch.arctanh(inputs * (2 - eps * 2) - 1 + eps)
    modifier = Variable(modifier, requires_grad=True)
    optimizer = optim.Adam([modifier], lr=learning_rate)

    lowFre_loss = nn.SmoothL1Loss(reduction='sum')
    highFre_loss = nn.SmoothL1Loss(reduction='sum')

    pbar = tqdm(range(num_iteration))
    for i in pbar:
        # for _ in range(num_iteration):
        optimizer.zero_grad()

        adv = 0.5 * (torch.tanh(modifier) + 1)  # [0, 1]
        adv_ll = DWT(adv)
        adv_hf = adv - IDWT(adv_ll)

        hf_cost = -0.1 * highFre_loss(adv_hf, inputs_hf)
        lf_cost = lowFre_loss(adv_ll, inputs_ll)
        # ta = model(inputs_turn.half().to(device)).latent_dist.mean
        adv_turn = adv * 2 - 1
        encoder_cost = 0.01 * (model(adv_turn.half().to(device)).latent_dist.mean - targets.detach()).norm()
        loss = lambda_sa * encoder_cost + lambda_fa * (lambda_hf * hf_cost + lambda_lf * lf_cost)
        # loss = lambda_hf * hf_cost + lambda_lf * lf_cost
        # print(loss)
        # print(f"loss: {loss:.4f}, encoder_loss: {encoder_cost:.4f}, hf_loss: {hf_cost:.4f}, lf_loss: {lf_cost:.4f}")
        pbar.set_description(f"[Running]: Loss {loss.item():.4f} | encLoss: {encoder_cost:.4f} | hfLoss: {hf_cost:.4f}")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            adv = 0.5 * (torch.tanh(modifier) + 1)
            perturbation = torch.clamp(adv - inputs, min=-epsilon / 2, max=epsilon / 2)
            modifier.data = torch.arctanh((inputs + perturbation) * (2 - eps * 2) - 1 + eps)

    adv = 0.5 * (torch.tanh(modifier.detach()) + 1)

    return adv


def get_target(model):
    # target_url = 'https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png'
    # target_url = 'https://pic.quanjing.com/ba/k3/QJ9110307926.jpg?x-oss-process=style/794ws'
    #response = requests.get(target_url)
    #target_image = Image.open(BytesIO(response.content)).convert("RGB")
    # target_image = target_image.resize((512, 512))
    # target_image = Image.open('./examples/image/example_3.png').convert("RGB")
    # target_image = target_image.resize((512, 512))
    target_image = Image.new("RGB", (512, 512), color="black")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_tensor = transform(target_image).unsqueeze(0).to(device)
    target = 2 * target_tensor - 1
    target = model(target.half().to(device)).latent_dist.mean
    return target

def tensor_to_image(tensor):
    return transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))


def main():
    print("start!")
    to_pil = T.ToPILImage()

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "./diffusion-v1-4/diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe_img2img = pipe_img2img.to("cuda")

    #image_folder = "./COCOEE/test_bench/GT_3500"
    # attack_folder = "./adv/tar_adv_005black"
    image_folder = "./imagenet/imagenet/imagenet/GT6"
    attack_folder = "./adv/tar_imagenet005"
    
    # image_files = os.listdir(image_folder)
    # adv_image_files = os.listdir(attack_folder)
    #image_folder = "./imagenet/copy"
    #attack_folder = "./adv/imgnet"
    image_files = os.listdir(image_folder)
    adv_image_files = os.listdir(attack_folder)
    # new_image_folder = "./imagenet/new"
    #  image_files = image_files[701:709]
    # image_files = image_files[50:1200]
    f = 0
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        filename = os.path.basename(image_path)[:8]
        # filename = os.path.basename(image_path)[16:24]
        # if any(filename == adv_file[:12] for adv_file in adv_image_files):
        #     continue
        adv_path = os.path.join(attack_folder, f"{filename}.png")
        # new_path = os.path.join(new_image_folder, f"{filename}.png")
        # image_path = "./autodl/examples/image/example_1.png"
        # targets = pipe_img2img.vae.encode(preprocess(get_target()).half().cuda()).latent_dist.mean
        targets = get_target(model=pipe_img2img.vae.encode)
        adversarial_sample = generate_adversarial_sample(image_path, targets=targets, model=pipe_img2img.vae.encode, )
        # if adversarial_sample == None:
        #     continue
        tensor_to_image(adversarial_sample).save(adv_path)
        # tensor_to_image(new).save(new_path)
        f += 1
        print(f"count: {f} finished!")

if __name__ == '__main__':
    main()
