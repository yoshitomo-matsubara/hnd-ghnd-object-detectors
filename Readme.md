HND y GHND para detectores de objetos
Destilación de red principal (HND) y HND generalizado para R-CNN más rápido, enmascarado y Keypoint

"Compresión y filtrado neuronales para la detección de objetos en tiempo real asistida por bordes en redes desafiadas", ICPR 2020
[ Preprint ]
"Computación dividida para detectores de objetos complejos: desafíos y resultados preliminares", Taller EMDL '20 de MobiCom 2020
[ PDF (acceso abierto) ] [ Preprint ]
GHND y filtro neural

Citas
@misc { matsubara2020neural ,
   title = { Compresión y filtrado neuronales para la detección de objetos en tiempo real asistida por bordes en redes desafiadas } ,
   autor = { Yoshitomo Matsubara y Marco Levorato } ,
   año = { 2020 } ,
   eprint = { 2007.15818 } ,
   archivePrefix = { arXiv } ,
   primaryClass = { cs.CV }
}

@inproceedings { matsubara2020split ,
   title = { Split Computing for Complex Object Detectors: Challenges and Preliminary Results } ,
   author = { Matsubara, Yoshitomo and Levorato, Marco } ,
   booktitle = { Proceedings of the 4th International Workshop on Embedded and Mobile Deep Learning } ,
   páginas = { 7--12 } ,
   año = { 2020 }
}
Requisitos
Python 3.6
pipenv
myutils
Cómo clonar
git clone https://github.com/yoshitomo-matsubara/hnd-ghnd-object-detectors.git
cd hnd-ghnd-object-detectors/
git submodule init
git submodule update --recursive --remote
pipenv install
No es necesario usar pipenv, y en su lugar puede instalar manualmente los paquetes requeridos enumerados en Pipfile , usando pip3.

Conjunto de datos COCO 2017
mkdir -p ./resource/dataset/coco2017
cd ./resource/dataset/coco2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
Puntos de control con pesos de modelo entrenados
Descarga emdl2020.zip aquí
Descomprima emdl2020.zip en el directorio raíz de este repositorio para que pueda usar los puntos de control con los archivos de configuración yaml en config / hnd /
Descarga icpr2020.zip aquí
Descomprima icpr2020.zip en el directorio raíz de este repositorio para que pueda usar los puntos de control con los archivos de configuración yaml en config / hnd / y config / ghnd /
Pruebe los modelos entrenados usando los puntos de control y los archivos de configuración yaml,
por ejemplo, Faster R-CNN con 3 canales de salida para el cuello de botella
pipenv run python src/mimic_runner.py --config config/hnd/faster_rcnn-backbone_resnet50-b3ch.yaml
pipenv run python src/mimic_runner.py --config config/ghnd/faster_rcnn-backbone_resnet50-b3ch.yaml
Porción de cabeza de destilación de R-CNN
Si ya ha descargado nuestros pesos de modelos entrenados arriba, debe mover los archivos ckpt resource/ckpt/a otro lugar o cambiar la ruta del archivo ckpt ( ckptdebajo student_model) en los archivos de configuración.

R-CNN más rápido inyectado en cuello de botella con ResNet-50 y FPN
por ejemplo, cuello de botella con 3 canales de salida

# HND
pipenv run python src/mimic_runner.py --config config/hnd/faster_rcnn-backbone_resnet50-b3ch.yaml -distill

# GHND
pipenv run python src/mimic_runner.py --config config/ghnd/faster_rcnn-backbone_resnet50-b3ch.yaml -distill
Mascarilla inyectada en cuello de botella R-CNN con ResNet-50 y FPN
por ejemplo, cuello de botella con 3 canales de salida

# HND
pipenv run python src/mimic_runner.py --config config/hnd/mask_rcnn-backbone_resnet50-b3ch.yaml -distill

# GHND
pipenv run python src/mimic_runner.py --config config/ghnd/mask_rcnn-backbone_resnet50-b3ch.yaml -distill
Keypoint R-CNN inyectado en cuello de botella con ResNet-50 y FPN
por ejemplo, cuello de botella con 3 canales de salida

# HND
pipenv run python src/mimic_runner.py --config config/hnd/keypoint_rcnn-backbone_resnet50-b3ch.yaml -distill

# GHND
pipenv run python src/mimic_runner.py --config config/ghnd/keypoint_rcnn-backbone_resnet50-b3ch.yaml -distill
Entrenamiento de un filtro neuronal sobre nuestro Keypoint R-CNN entrenado e inyectado en cuellos de botella
pipenv run python src/ext_runner.py --config config/ext/keypoint_rcnn-backbone_ext_resnet50-b3ch.yaml -train
Referencias
pytorch / visión / referencias / detección /
código para visualización en el tutorial de detección de objetos
