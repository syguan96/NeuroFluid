# NeuroFluid
Code repository for this paper:  
**NeuroFluid: Fluid Dynamics Grounding with Particle-Driven Neural Radiance Fields.**  
[Shanyan Guan](https://syguan96.github.io/), Huayu Deng, [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)  
ICML 2022   
[[Paper]](https://arxiv.org/pdf/2203.01762.pdf) [[Project Page]](https://syguan96.github.io/NeuroFluid/)

Please cite our paper if you find this code useful:
```
@inproceedings{guan2022neurofluid,
  title={NeuroFluid: Fluid Dynamics Grounding with Particle-Driven Neural Radiance Fields},
  author={Guan, Shanyan and Deng, Huayu and Wang, Yunbo and Yang, Xiaokang},
  booktitle={ICML},
  year={2022}
}
```


<p float="center">
  <img src="https://github.com/syguan96/ImageHost/blob/main/teaser.gif" width="99%" />
</p>

## Dependencies
NeuroFluid is implemented and tested on Ubuntu 18.04 with python == 3.7 and cuda == 11.1. To run NeuroFluid, please install dependencies as follows:
1. Create an environment
    ```bash 
    conda create -n fluid-env python=3.7
    conda activate fluid-env
    ```
2. Install Open3D.   
    ```bash
    git clone https://github.com/isl-org/Open3D-ML.git && cd Open3D-ML && git checkout c461790869257e851ae7f035585b878df73bc093
    pip install open3d==0.15.2
    pip install -r requirements.txt
    pip install -r requirements-torch-cuda.txt
    ```
3. Install Pytorch3D
    ```bash
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    wget -c https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.6.1.tar.gz
    tar -xf v0.6.1.tar.gz && cd pytorch3d-0.6.1 && pip install -e .
    ```
4. install other dependencies
    ```bash
    pip install -r requirements.txt
    ```



## Generate data
See [this guide](data_generation/README.md) to generate fluid data.

## Running the pretrained model
- Evaluate NeuroFluid:
```bash 
python eval_e2e.py --resume_from $MODEL_PATH --dataset DATASET_NAME
```
DATASET_NAME is one element of [bunny, watercube, watersphere, honeycone]. To compute PSNR/SSIM/LPIPS results, run `utils/evaluate_images.ipynb` 

- Evaluate the transition model
```bash
python eval_transmodel.py --resume_from $MODEL_PATH
```

- Evaluate the renderer
```bash
python eval_renderer.py --resume_from $MODEL_PATH --dataset DATASET_NAME
```
DATASET_NAME is one element of [bunny, watercube, watersphere, honeycone].



## Run the training script
1. Warm-up stage
```bash
CUDA_VISIBLE_DEVICES=3 python train_renderer.py --expdir exps/watercube --expname scale-1.0/warmup --dataset watercube --config configs/watercube_warmup.yaml 
```

2. End2end stage
```bash 
CUDA_VISIBLE_DEVICES=0 python train_e2e.py --expdir exps/watercube --expname scale-1.0/e2e --dataset watercube --config configs/watercube_e2e.yaml 
```

## Fetch data
Download dataset and models from this [link](https://drive.google.com/drive/folders/14lOFgqE2XMhFrJUj94IQoIVa2q1R0v2C?usp=sharing).

## Acknowledgement
The implementation of transition model is borrowed from [
DeepLagrangianFluids](https://github.com/isl-org/DeepLagrangianFluids). Please consider cite their paper if you use their code snippet:
```
@inproceedings{Ummenhofer2020Lagrangian,
        title     = {Lagrangian Fluid Simulation with Continuous Convolutions},
        author    = {Benjamin Ummenhofer and Lukas Prantl and Nils Thuerey and Vladlen Koltun},
        booktitle = {International Conference on Learning Representations},
        year      = {2020},
}
```

We refer to [nerf_pl](https://github.com/kwea123/nerf_pl) to implement our renderer. Thank D-NeRF for providing the script of computing PSNR/SSIM/LPIPS.
