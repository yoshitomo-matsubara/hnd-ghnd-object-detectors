# Distilled Object Detector

## Requirements
- Python 3.6
- pipenv
- [myutils](https://github.com/yoshitomo-matsubara/myutils)

## How to clone
```
git clone https://github.com/yoshitomo-matsubara/distilled-object-detector.git
cd distilled-object-detector/
git submodule init
git submodule update --recursive --remote
pipenv install
```

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
