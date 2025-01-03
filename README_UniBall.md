# Data preprocess

1. put original json annotation files into

   ```
   data/annotations_raw/
   data/video_raw/
   ```

   the structure should be

   ```
     data
       └── annotations_raw
           ├── video_raw/
           │   └── tabletennis/
           │       ├── 0lvsnujz_000.mp4
           │       ├── 0lvsnujz_001.mp4
           │       ├── …
           │       └── zhwjr64d_019.mp4
           └── annotations_raw/
               └── tabletennis/
                   └── 0lvsnujz_000
                       ├── 0009.jpg.json
                       ├── 0018.jpg.json
                       ├── ...
                       └── 0240.jpg.json
   ```
2. ```
   python preprocess_extract_frames.py
   python preprocess_collect_annotation.py
   python preprocess_split_train_val_test.py
   python preprocess_create_median.py

   ```
3. the generated dataset structure should be

   ```
   data
   └── tabletennis
       ├── all
       │   ├── match1
       │   │   ├── csv
       │   │   ├── frame
       │   │   └── median.npz
       │   ├── ...
       │   └── match9
       │       ├── csv
       │       ├── frame
       │       └── median.npz
       ├── info
       │   ├── metainfo.json
       │   ├── test.json
       │   ├── train.json
       │   └── val.json
       ├── match_to_idxs.json
       └── median
           ├── match10_median.png
           ├── ...
           └── match9_median.png
   ```

# Installation

- Install PyTorch 2.1.1+cu118
  ```shell
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
  ```
- Install `causal_conv1d` and `mamba`
  ```shell
  pip install -r requirements.txt
  pip install -e causal-conv1d
  pip install -e mamba
  ```
