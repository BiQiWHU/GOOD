# [ISPRS'2025] ```GOOD:Towards Domain Generalized Oriented Object Detection```

This is the official implementation of our work entitled as ```GOOD:Towards Domain Generalized Oriented Object Detection```, which has been accepted by ```ISPRS'2025```.

## Environment Configuration

```
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
``` 

## Training

On a single GPU

```
python tools/train.py ${CONFIG_FILE} [optional arguments]
``` 

On multiple GPUs

```
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

For the details of ```config file```, please refer to:

*[https://github.com/open-mmlab/mmrotate/blob/main/configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py]

*[https://github.com/open-mmlab/mmrotate/blob/main/configs/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90.py]

## Inference

On a single GPU

```
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
``` 

On multiple GPUs

```
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
``` 

## Citation

If you find this work benefits your research, please cite our work as follows:

```BibTeX
@article{bi2025good,
  title={GOOD: Towards domain generalized oriented object detection},
  author={Bi, Qi and Zhou, Beichen and Yi, Jingjun and Ji, Wei and Zhan, Haolan and Xia, Gui-Song},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={223},
  pages={207--220},
  year={2025}
}
```

## Acknowledgement

Our implementation is primarily based on ```mmrotate```. Thanks for their authors.
* [mmrotate](https://github.com/open-mmlab/mmrotate)
