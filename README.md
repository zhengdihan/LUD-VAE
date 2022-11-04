# LUD-VAE
Official code for our paper "Learn from Unpaired Data for Image Restoration: A Variational Bayes Approach". https://ieeexplore.ieee.org/document/9924527/

## Dataset Preparation
For AIM19 and NTIRE20, the dataset preparation is the same with the DeFlow method. See https://github.com/volflow/DeFlow.

For SIDD, we use the SIDD-Small Dataset, which can be download from https://www.eecs.yorku.ca/~kamel/sidd/dataset.php.
We crop the images in SIDD-Small dataset to 512x512x3 patches.

For DND, the dataset can be find in https://www.bing.com/search?q=dnd+dataset&cvid=7faaa429db2e4c28a33a4fd35fffe0ec&aqs=edge.0.0l2.3624j0j1&pglt=43&FORM=ANNTA1&PC=EDGEDSE

For LOL, we use the EnlGAN's unpaired dataset: https://drive.google.com/drive/folders/1fwqz8-RnTfxgIIkebFG2Ej3jQFsYECh0?usp=sharing
and the test data from LOL dataset: https://daooshee.github.io/BMVC2018website/

##

## Validate Pretrained Models

We provide the pre-trained LUD-VAE model in `*/trained_models/LUDVAE_models/ludvae.pth`, and a demo test for validate our model, run:

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
We provide pre-trained models in 
```
./LUD_VAE_dnd/DnCNN/model_zoo/dncnn.pth 
```
and 
```
./LUD_VAE_sidd/DnCNN/model_zoo/dncnn.pth 
```
for DND and SIDD datasets.

For low-light image enhancement task, we use the LLFlow model: https://github.com/wyf0912/LLFlow

All the pretrained models can be found in https://drive.google.com/drive/folders/1XKIt5WBPh0Fc3AFEcWxZphfs_N_5YnYw?usp=sharing

## Training LUD-VAE

Change the `"dataroot"` in `*/options/train_*.json`, run:

```
cd ./LUD_VAE_aim19/
python main_train.py
```
