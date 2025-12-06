# Public Image Deepfake Detectors Adapter

## Usage


```bash
python detect.py \
  --folders <FOLDER_1> <FOLDER_2> ... \
  --limit <N> \
  --models [all | CLIP-D | NPR | P2G | R50_nodown | R50_TF | WaveRep] \
  --weights <PATH> \
  --output <results.csv> \
  --device <DEVICE> \
  --batch_size <BATCH_SIZE>
```