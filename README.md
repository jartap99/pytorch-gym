# pytorch-gym

All experiments are on GTX 1070 with WSL=2 on Ubuntu 20.04

```
$ conda env create --file=environment.yaml
$ pip install torchsummary
$ pip install flask
$ pip install captum
```

For tensorboard, run tensorboard in background
```
$ tensorboard --logdir=<parent dir>
```

For vision Mask R-CNN
```
$ pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
based on https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=DBIoe_tHTQgV
