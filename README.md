## MoNet: Motion-based Point Cloud Prediction Network

### Environments

- PyTorch 1.7.1
- cuda 11.1
- [pytorch3d](https://github.com/facebookresearch/pytorch3d)
- [EMD](https://github.com/daerduoCarey/PyTorchEMD)

Please run the following commands to install `point_utils`
```
cd model/PointUtils
python setup.py install
```

Please check `requirements.txt` for more requirements.

### Datasets
The data of the two datasets should be organized as follows:
#### KITTI odometry dataset
```
DATA_ROOT
├── 00
│   ├── velodyne
│   ├── calib.txt
├── 01
├── ...
```
#### Argoverse dataset
```
DATA_ROOT
├── train1
│   ├── 043aeba7-14e5-3cde-8a5c-639389b6d3a6
|       ├──lidar
|       ├──poses
|       ├──...
│   ├── ...
├── train2
├── train3
├── train4
├── val
├── test
```

### Evaluation

Please run `eval_kitti.sh/eval_argo.sh` to evaluate the proposed MoNet on the two datasets using the provided pretrained model in `ckpt`. The `ROOT`, `CKPT`, `GPU` and `RNN` should be modified.

### Train

If you want to train the network, please run `train.sh` and reminder to modify the `ROOT`, `CKPT_DIR` and `RUNNAME`. 

Noting that we utilize [wandb](https://www.wandb.com/) to record the training procedure, if you do not want to use it, please drop the `--use_wandb` in `train.sh`.

### Citation
If you find this project useful for your work, please consider citing:
```
@ARTICLE{Lu_MoNet_2021,
    author={Lu, Fan and Chen, Guang and Li, Zhijun and Zhang, Lijun and Liu, Yinlong and Qu, Sanqing and Knoll, Alois},
    journal={IEEE Transactions on Intelligent Transportation Systems},
    title={MoNet: Motion-Based Point Cloud Prediction Network}, 
    year={2021},
    volume={},
    number={},
    pages={1-11}
}
```