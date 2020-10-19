# HND & GHND for Object Detectors
Head Network Distillation (HND) and Generalized HND for Faster, Mask, and Keypoint R-CNNs  
- "Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks," [ICPR 2020](https://www.micc.unifi.it/icpr2020/)  
[[Preprint](https://arxiv.org/abs/2007.15818)]
- "Split Computing for Complex Object Detectors: Challenges and Preliminary Results," [MobiCom 2020 Workshop EMDL '20](https://emdl20.github.io/index.html)  
[[PDF (Open Access)](https://dl.acm.org/doi/abs/10.1145/3410338.3412338)] [[Preprint](https://arxiv.org/abs/2007.13312)]


## Citations
```bibtex
@misc{matsubara2020neural,
  title={Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks},
  author={Yoshitomo Matsubara and Marco Levorato},
  year={2020},
  eprint={2007.15818},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{matsubara2020split,
  title={Split Computing for Complex Object Detectors: Challenges and Preliminary Results},
  author={Matsubara, Yoshitomo and Levorato, Marco},
  booktitle={Proceedings of the 4th International Workshop on Embedded and Mobile Deep Learning},
  pages={7--12},
  year={2020}
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

## COCO 2017 Dataset
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
1. Download **emdl2020.zip** [here](https://drive.google.com/file/d/1lMwkaoxhnw260FgZoTLvWRZXajlSxKtN/view?usp=sharing)
2. Unzip **emdl2020.zip** at the root directory of this repository so that you can use the checkpoints with yaml config files under [config/hnd/](config/hnd/)
3. Download **icpr2020.zip** [here](https://drive.google.com/file/d/1K7MNVuW99uDMHciewVS71hks_YdU9_2A/view?usp=sharing)
4. Unzip **icpr2020.zip** at the root directory of this repository so that you can use the checkpoints with yaml config files under [config/hnd/](config/hnd/) and [config/ghnd/](config/ghnd/)
5. Test the trained models using the checkpoints and yaml config files  
e.g., Faster R-CNN with 3 output channels for bottleneck
```
pipenv run python src/mimic_runner.py --config config/hnd/faster_rcnn-backbone_resnet50-b3ch.yaml
pipenv run python src/mimic_runner.py --config config/ghnd/faster_rcnn-backbone_resnet50-b3ch.yaml
```

## References
- [pytorch/vision/references/detection/](https://github.com/pytorch/vision/tree/master/references/detection)
- [code for visualization in the object detection tutorial](https://github.com/pytorch/vision/issues/1610)
