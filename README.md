# HND & GHND for Object Detectors
Head Network Distillation (HND) and Generalized HND for Faster, Mask, and Keypoint R-CNNs

## Citations
```bibtex
@article{matsubara2020split,
    title={Split Computing for Complex Object Detectors: Challenges and Preliminary Results},
    author={Matsubara, Yoshitomo and Levorato, Marco},
    year={2020},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.ML}
}
```

## Requirements
- Python 3.6
- pipenv
- [myutils](https://github.com/yoshitomo-matsubara/myutils)

## How to clone
```
git clone https://github.com/yoshitomo-matsubara/hnd-ghnd-object-detectors.git
cd hnd-ghnd-object-detectors/
git submodule init
git submodule update --recursive --remote
pipenv install
```
It is not necessary to use pipenv, and you can instead manually install the required packages listed in [Pipfile](Pipfile), using pip3

## Dataset
COCO 2017
```
mkdir -p ./resource/dataset/coco2017
cd ./resource/dataset/coco2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
```

## Checkpoints with trained model weights
1. Download **emdl2020.zip** [here](https://drive.google.com/file/d/1l1RHT_BhTJ_yh-z5L4x04Nj2-Oe4w9C1/view?usp=sharing)
2. Unzip **emdl2020.zip** at the root directory of this repository so that you can use the checkpoints with yaml config files under [config/hnd/](config/hnd/)
3. Test the trained models using the checkpoints and yaml config files  
e.g., Faster R-CNN with 3 output channels for bottleneck
```
pipenv run python src/coco_runner.py --config config/hnd/faster_rcnn-backbone_resnet50-b3ch.yaml
```

## References
- [pytorch/vision/references/detection/](https://github.com/pytorch/vision/tree/master/references/detection)
- [code for visualization in the object detection tutorial](https://github.com/pytorch/vision/issues/1610)
