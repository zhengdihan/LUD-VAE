# LUD-VAE
Official code for "Learn from Unpaired Data for Image Restoration: A Variational Bayes Approach".

## Dataset Preparation
For AIM19 and NTIRE20, the dataset preparation is the same with the DeFlow method. See https://github.com/volflow/DeFlow.

For SIDD, we use the SIDD-Small Dataset, which can be download from https://www.eecs.yorku.ca/~kamel/sidd/dataset.php.

## Validate Pretrained Models

We provide the trained LUD-VAE model in `*/trained_models/LUDVAE_models/ludvae.pth`, and a demo test for validate our model, run:

```
cd ./LUD_VAE_aim19/
python main_test.py
```

## Generate Synthetic Datasets

To generate synthetic datasets, change the `H_path` in `*/main_translate.py`, and run:
```
python main_translate.py
```

## Downstream Models

For real-world super-resolution, we use the ESRGAN model. The code can be found in https://github.com/jixiaozhong/RealSR.

For image denoising task, we use the DnCNN model, which from https://github.com/cszn/KAIR.
