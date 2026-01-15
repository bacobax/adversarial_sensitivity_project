# Image Deepfake Detectors Adapter

## Clone the Repo

Use the following command to clone only the latest version and avoid heavy history and branches:
```bash
git clone --depth 1 --branch main --single-branch https://github.com/bacobax/adversarial_sensitivity_project.git
```

## Base Repository

- [Image Deepfake Detectors Public Library](https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library)

## Models

- [R50_nodown](https://grip-unina.github.io/DMimageDetection)
- [WaveRep](https://grip-unina.github.io/WaveRep-SyntheticVideoDetection/)
- [AnomalyOV](https://github.com/honda-research-institute/Anomaly-OneVision)

## Download Weights

You can download the weights for each model from this [link](https://drive.google.com/file/d/1F60FN2B9skRcb3YrZwhFTZQihbj3ipJQ/view?usp=sharing).

Then, copy them into the `weights` folder for the corresponding model, following this structure:
`./models/<DETECTOR>/weights/best.pt`

Download also the classes for P2G detector from [classes.pkl](https://github.com/laitifranz/Prompt2Guard/blob/main/src/utils/classes.pkl) and copy it into the `models/P2G/utils` folder.

## Dataset

The dataset used for evaluation is [B-Free](https://github.com/grip-unina/B-Free), downloaded from [here](https://www.grip.unina.it/download/prog/B-Free/training_data/).

## Usage

```bash
python3.13 detect.py \
    --folders <path/to/folder1> <path/to/folder2> ... \
    [--detectors all | CLIP-D NPR P2G R50_nodown R50_TF WaveRep ...] \
    [--weights <model1:/path/to/weights> <model2:/path/to/weights> ...] \
    [--limit 0 | <max images per folder>] \
    [--device cuda:0 | cpu] \
    [--batch_size 16] \
    [--output results.csv]
```

## Requirements

```bash
conda env create -f environment.yml
```
