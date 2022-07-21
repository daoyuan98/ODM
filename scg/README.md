# Spatially Conditioned Graphs
## Note: this is mainly provided by the anthors of SCG

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Utilities](#data-utilities)
    * [HICO-DET](#hico-det)
- [Testing](#testing)
    * [HICO-DET](#hico-det-1)
- [Training](#training)
    * [HICO-DET](#hico-det-2)
- [Contact](#contact)

## Prerequisites

1. Download the repository with `git clone https://github.com/fredzzhang/spatially-conditioned-graphs`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
3. Make sure the environment you created for Pocket is activated. You are good to go!


## Data Utilities

The [HICO-DET](https://github.com/fredzzhang/hicodet) and [V-COCO](https://github.com/fredzzhang/vcoco) repos have been incorporated as submodules for convenience. To download relevant data utilities, run the following commands.
```bash
cd /path/to/spatially-conditioned-graphs
git submodule init
git submodule update
```
### HICO-DET
1. Download the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/spatially-conditioned-graphs/hicodet
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link
    ```bash
    cd /path/to/spatially-conditioned-graphs/hicodet
    ln -s /path/to/hico_20160224_det ./hico_20160224_det
    ```
2. Run a Faster R-CNN pre-trained on MS COCO to generate detections
```bash
cd /path/to/spatially-conditioned-graphs/hicodet/detections
python preprocessing.py --partition train2015
python preprocessing.py --partition test2015
```
3. Generate ground truth detections (optional)
```bash
cd /path/to/spatially-conditioned-graphs/hicodet/detections
python generate_gt_detections.py --partition test2015 
```
4. Download fine-tuned detections (optional)
```bash
cd /path/to/spatially-conditioned-graphs/download
bash download_finetuned_detections.sh
```
To attempt fine-tuning yourself, refer to the [instructions](https://github.com/fredzzhang/hicodet/tree/main/detections#fine-tune-the-detector-on-hico-det) in the [HICO-DET repository](https://github.com/fredzzhang/hicodet). The checkpoint of our fine-tuned detector can be found [here](https://drive.google.com/file/d/11lS2BQ_In-22Q-SRTRjRQaSLg9nSim9h/view?usp=sharing).


## Testing
### HICO-DET
1. Download the checkpoint of our trained model
```bash
cd /path/to/spatially-conditioned-graphs/download
bash download_checkpoint.sh
```
2. Test a model
```bash
cd /path/to/spatially-conditioned-graphs
CUDA_VISIBLE_DEVICES=0 python test.py --model-path checkpoints/scg_1e-4_b32h16e7_hicodet_e2e.pt
```
By default, detections from a pre-trained detector is used. To change sources of detections, use the argument `--detection-dir`, e.g. `--detection-dir hicodet/detections/test2015_gt` to select ground truth detections. Fine-tuned detections (if you downloaded them) are available under `hicodet/detections`.

3. Cache detections for Matlab evaluation following [HO-RCNN](https://github.com/ywchao/ho-rcnn) (optional)
```bash
cd /path/to/spatially-conditioned-graphs
CUDA_VISIBLE_DEVICES=0 python cache.py --model-path checkpoints/scg_1e-4_b32h16e7_hicodet_e2e.pt
```
By default, 80 `.mat` files, one for each object class, will be cached in a directory named `matlab`. Use the `--cache-dir` argument to change the cache directory. To change sources of detections, refer to the use of `--detection-dir` in the previous section.


## Training
### HICO-DET
```bash
cd /path/to/spatially-conditioned-graphs
python main.py --world-size 8 --cache-dir checkpoints/hicodet &>log &
```
Specify the number of GPUs to use with the argument `--world-size`. The default sub-batch size is `4` (per GPU). The provided model was trained with 8 GPUs, with an effective batch size of `32`. __Reducing the effective batch size could result in slightly inferior performance__. The default learning rate for batch size of 32 is `0.0001`. As a rule of thumb, scale the learning rate proportionally when changing the batch size, e.g. `0.00005` for batch size of `16`. It is recommended to redirect `stdout` and `stderr` to a file to save the training log (as indicated by `&>log`). To check the progress, run `cat log | grep mAP`, or alternatively you can go through the log with `vim log`. Also, the mAP logged follows a slightly different protocol. It does __NOT__ necessarily correlate with the mAP that the community reports. It only serves as a diagnostic tool. The true performance of the model requires running a seperate test as shown in the previous section. By default, checkpoints will be saved under `checkpoints` in the current directory. For more arguments, run `python main.py --help` to find out. We follow the early stopping training strategy, and have concluded (using a validation set split from the training set) that the model at epoch `7` should be picked. Training on 8 GeForce GTX TITAN X devices takes about `5` hours.
