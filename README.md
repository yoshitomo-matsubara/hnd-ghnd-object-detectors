# HND & GHND for Object Detectors

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

@article{matsubara2020neural,
    title={Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks},
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

## References
- [pytorch/vision/references/detection/](https://github.com/pytorch/vision/tree/master/references/detection)
- [code for visualization in the object detection tutorial](https://github.com/pytorch/vision/issues/1610)
