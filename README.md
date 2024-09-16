# PSFD_code

（The code will continue to be organised as well as improved, and the full code will be open-sourced after acceptance of the paper.）
## Description of the document

PSFD-T.py and PSFD-U: Generate corresponding adversarial samples and adjust the noise size by adjusting epsilon.

scripts: Contains code for image editing and some examples

v1.yaml：configuration file

## code running

#### Generate corresponding adversarial samples

```shell1
python PSFD-T.py or PSFD-U.py
```

#### image editing

```shell
python scripts/inference_1.py \
--plms --outdir your_results_path \
--config configs/v1.yaml \
--ckpt /HARD-DATA/ZHT/.cache/huggingface/hub/models--Fantasy-Studio--Paint-by-Example/snapshots/351e6427d8c28a3b24f7c751d43eb4b6735127f7/model.ckpt
```

Make sure you have the checkpoint model installed!（stable-diffusion-v1-4)

