Umgebung einrichten (Optimalerweise in virtueller Umgebung):


frankfurt_000001_055387_gtFine_labelIds

In meinem Falle mit
- Python 3.9
- CUDA 11.7

Folgendes im Terminal ausführen:


pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt


Damit sollte alles installiert sein.


Testen mit Inference:

python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none

Auf Marius' Shrottmühle:
python test.py --name label2city_1024p


python test2.py --name label2city_1024p --no_instance --label_nc 0 --resize_or_crop crop --fineSize 1024


Zum Training mit voller Auflösung (min. VRAM 24GB, 16GB mit "mixed precision"):

python train.py --name label2city_1024p --label_nc 0 --no_instance



Mit 512p:

python train.py --name label2city_512p





Referenzen:

https://github.com/NVIDIA/pix2pixHD/pull/204/commits/8b184387940650edb70edc5b86c0cf86d8e6b4e4
https://github.com/NVIDIA/pix2pixHD/pull/271/commits/1414723838842bbcdc04d057a077e1fd08a89aa4