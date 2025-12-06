# Image Deepfake Detectors Adapter

## Base Repository

- [Image Deepfake Detectors Public Library](https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library)

## Models

- [CLIP-D](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)
- [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [P2G](https://github.com/laitifranz/Prompt2Guard)
- [R50_nodown](https://grip-unina.github.io/DMimageDetection)
- [R50_TF](https://github.com/MMLab-unitn/TrueFake-IJCNN25)
- [WaveRep](https://grip-unina.github.io/WaveRep-SyntheticVideoDetection/)

## Usage

```bash
python3.13 detect.py \
    --folders <path/to/folder1> <path/to/folder2> ... \
    [--models all | CLIP-D NPR P2G R50_nodown R50_TF WaveRep ...] \
    [--weights <model1:/path/to/weights> <model2:/path/to/weights> ...] \
    [--limit 0 | <max images per folder>] \
    [--device cuda:0 | cpu] \
    [--batch_size 16] \
    [--output results.csv]
```

## Requirements

```bash
python3.13 -m pip install -r requirements.txt
```

```bash
conda env create -f environment.yml
```