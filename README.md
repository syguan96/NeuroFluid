# NeuroFluid
Code reposity for this paper:  
**NeuroFluid: Fluid Dynamics Grounding with Particle-Driven Neural Radiance Fields.**  
[Shanyan Guan](https://syguan96.github.io/), Huayu Deng, [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)  
ICML 2022   
[[Paper]](https://github.com/syguan96/NeuroFluid)/[[Project Page]](https://syguan96.github.io/NeuroFluid/)

Please cite our paper (pdf) if you find this code useful:
```
@inproceedings{guan2022neurofluid,
  title={NeuroFluid: Fluid Dynamics Grounding with Particle-Driven Neural Radiance Fields},
  author={Guan, Shanyan and Deng, Huayu and Wang, Yunbo and Yang, Xiaokang},
  booktitle={ICML},
  year={2022}
}
```



## Dependencies
NeuroFluid is implemented and tested on Ubuntu 18.04 with python == 3.7. To run NeuroFluid, please install dependencies as follows:
1. Create an environment
    ```bash 
    conda create -n fluid-env python=3.7
    conda activate fluid-env
    ```
2. Install Open3D.   
    ```bash
    git clone https://github.com/isl-org/Open3D-ML.git
    cd Open3D-ML
    pip install open3d
    pip install -r requirements.txt
    pip install -r requirements-torch-cuda.txt
    ```
3. Install Pytorch3D
    ```bash
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d && pip install -e .
    ```
4. install other dependencies
    ```bash
    pip install -r requirements.txt
    ```



## Fetch data
To be down.



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
To be down.



## Evaluation on baselines
To be down.

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

We refer to [nerf_pl](https://github.com/kwea123/nerf_pl) to implement our renderer.