# Dust spot segmentations


## Table of Contents

- [Dust spot segmentations](#dust-spot-segmentations)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Usage](#usage)
    - [Before starting](#before-starting)
    - [Data preparation](#data-preparation)
    - [Training](#training)
    - [Inference](#inference)
  - [Dataset collection, Possible improvements and Alternative methods](#dataset-collection-possible-improvements-and-alternative-methods)


## Description
Task - to create a model for detection of dust spot on the high resolution picture.

I propose the following algorithm for solving this problem:

1. Resize very-high-res picture to `2016x2016` pixels.
2. Сut the image into 81 pieces with the shape `224x224` pixels.
3. Run segmentation model Unet (with EfficientNet-B1 backbone) on all those images. This means, run 9 batches with the shape `(9, 3, 224, 224)`. This will give us a segmentation mask for each of the 81 pieces.
4. Merge all 81 pieces into one big `2016x2016` segmentation mask.

The image processing algorithm (image cutting + model inference + image merging) works with images with a resolution of `2016x2016` pixels on a **MacBook Pro (13-inch, 2019) with 2.4 GHz Quad-Core Intel Core i5** processor at average 1.7-2.3 seconds per image. Logs:

```
Img time: 2.3120357990264893
Img time: 1.7598841190338135
Img time: 1.775665044784546
Img time: 1.7746961116790771
Img time: 1.752397060394287
Img time: 1.791938066482544
Img time: 1.890420913696289

Results:
Timer: 30 sec.
Number of processed images: 7
Mean time for pipeline: 1.8652910164424352
```

Algorithm was tested with a small benchmark, which can be run in the `inference.py` file.

Unet model was designed in `PyTorch` and optimized and converted from `PyTorch` FP32 precision into `OpenVINO` FP16 precision. And the inference pipeline works on OpenVINO backend. All converted models from benchmark can be found in `experiments/bechmark` folder after running the `inference.py` in _benchmark mode_.

Useful structure info:
1. Segmentation model can be found in `model/unet.py`.
2. Train pipeline can be found in `train.py`
2. Inference pipeline can be found in `inference.py`
3. Algorithms for image cutting and merging can be found in `utils.py`
4. Data preparation scripts can be found in `dataset_preparation.py`.
5. Model optimization and conversion scripts can be found in `model_convert.py`


## Usage

### Before starting
Run in terminal:
```
virtualenv venv -p python3.9
source venv/bin/activate
pip install -r requirements.txt
```

### Data preparation

Firstly, for model training, you need labeled data. `Dataset` uses `Segmentation mask 1.1` format of labels from CVAT. After labeling, split your data into `train` and `valid` folder. 

Run `dataset_preparation.py` to cut all you images from the dataset:
```
python dataset_preparation.py --dataset_path path/to/your/data --export_dir path/for/export
```

If the `export_dir` directory does not exist, it will be created automatically. Also, you can provide `big_size` and `piece_size` params from `CFG` class in `config.py`

### Training

All training parameter can be found and changed in in `config.py`. Before training, you have to change `train_data` and `valid_data` atributes and provide here paths to the datasets with cutted images.

Run training:
```
python train.py
```

### Inference

`inference.py` file have 2 modes: benchmark (default) and regular inference.

Benchmark mode uses only noisy data and runs simple with:
```
python inference.py --benchmark True --timer 60
```

This will run benchmark with 60-second timer.

Regular inference requires `dataset_path`, `export_dir` and `model_path`:
```
python inference.py --benchmark False --dataset_path path/to/your/data --export_dir path/for/export/masks --model_path path/to/model.pth
```

## Dataset collection, Possible improvements and Alternative methods

For dataset collection I will provide the following steps:

1. Try to generate small/mid size dataset with  synthetic dust spots on real images and train first iteration of the model on this data.
2. Start web scrapping for high-res images and and select those images in which the trained model has found something. This method gives us data with TP and FP. Select about 1000-2000 images.
3. Start labeling this selected images. Create traing and test datasets.
4. Repeat from step 2 few times.


For possible improvement I can suggest several possible improvements to this solution:

1. More gently setup training params and augmentations.
2. Add an extra check at the joints of the pieces, if there is a dust spot at these joints
3. Try to use `einops.rearrange` for image cutting if it will give performance Improvements.
4. Convert model into `INT8` format. This solution  [requers Calibration Dataset](https://docs.openvino.ai/latest/notebooks/114-quantization-simplified-mode-with-output.html#compression-stage).
5. Try [Model Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#iterative-pruning) in order to accelerate inference pipeline.
6. Implement simple inpainting techniques, for example [with OpenCV](https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html) [[documentation](https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085)].
7. Provide deployment pipeline with `Docker` or/and `Redis`


During my research I found few possible solutions for this and neighboring tasks:
1. Use segmentations model ([ref](https://github.com/videogorillas/pytorch-unet-dust)) or detection models ([ref](https://www.nature.com/articles/s41598-022-19674-8))
2. Use GANs ([ref](https://arxiv.org/pdf/2009.10663.pdf)) or Transformers ([ref](https://arxiv.org/pdf/2012.00364v4.pdf)). But it is almost impossible to effectively run such large models on CPUs. And these models can produce artifacts because they fully regenerate images.
3. Use Variational Autoencoders (VAE) ([ref](https://users.wpi.edu/~yli15/Includes/AdvML20_Huimin.pdf)). But they also generate a lot of artifacts.