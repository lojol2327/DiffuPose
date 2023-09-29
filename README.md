# DiffuPose: Monocular 3D Human Pose Estimation via Denoising Diffusion Probabilistic Model

![S11_Walking](https://user-images.githubusercontent.com/54428449/202689921-ca7306e7-0754-4319-9e93-5cf988147fb9.gif)

## Environment
This code is built on the following environment
* Python 3.8.13
* PyTorch 1.12.1
* CUDA 11.2

You can create and activate conda environment as the following:
```
conda env create -f requirements.yaml
conda activate diffupose
```

# Dataset setup
You can download 2D HR-Net detection, 2D ground-truth and 3D ground-truth on Human3.6M dataset via our [Google Drive](https://drive.google.com/file/d/12A3PCMrr5DvFjlYMUUZFqDaSvhdtlfm5/view?usp=share_link) link. (It is only a temporary link for review).
Download them and unzip the .zip file in the ./data folder.

You will end up with './data/data_3d_h36m.npz', './data/data_2d_h36m_hr.npz', './data/data_2d_h36m_gt.npz'.

# Training from scratch
If you want to train our model from scratch using HR-Net detection, please run
```
python run.py -k hr -b 1024
```
Else, if you want to train with 2D ground-truth, please run
```
python run.py -k gt -b 1024
```

# Evaluating pre-trained model
We provide our pre-trained 384-dimension model in results folder (HR-Net detected 2D pose as input). To evaluate our model, please run
```
python run.py -k hr --test-load best_model.pt
```
which will result in 50.0 mm error (MPJPE).

To obatin best result with 10 samples, run
```
python run.py -k hr --test-load best_model.pt --num-sample 10
```
which will result in 49.4 mm error (MPJPE).

# Visualization
Our code is compatiable with [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to their github page for detailed instruction.
