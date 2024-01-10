This repository is folked from [https://github.com/MrParosk/ml_playground](https://github.com/MrParosk/ml_playground).
At this repository, simplification and explanation and will be tested at Colab Environment.

## SSD

#### Engilish
*  **Theory** : [https://wikidocs.net/226339](https://wikidocs.net/226339) <br>
*  **Implementation** : [https://wikidocs.net/226340](https://wikidocs.net/226340)

#### 한글
*  **Theory** : [https://wikidocs.net/204452](https://wikidocs.net/204452) <br>
*  **Implementation** : [https://wikidocs.net/225903](https://wikidocs.net/225903)

## Retinanet

#### Engilish
*  **Theory** : [https://wikidocs.net/227235](https://wikidocs.net/227235) <br>
*  **Implementation** : [https://wikidocs.net/227236](https://wikidocs.net/227236)

#### 한글
*  **Theory** : [https://wikidocs.net/225902](https://wikidocs.net/225902) <br>
*  **Implementation** : [https://wikidocs.net/225930](https://wikidocs.net/225930)


## Faster R-CNN
#### Engilish
*  **Theory** : [https://wikidocs.net/204455](https://wikidocs.net/204455) <br>
*  **Implementation** : [https://wikidocs.net/226581](https://wikidocs.net/226581)

#### 한글
*  **Theory** : [https://wikidocs.net/204453](https://wikidocs.net/204453) <br>
*  **Implementation** : [https://wikidocs.net/226579](https://wikidocs.net/226579)


## Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_J_K_Pytorch_SSD_Retinanet_Faster_R_CNN.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

# Training and test
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

# ! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# ! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# !tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
# !tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
# clear_output()
```

## Data convert to json files

```
! python src/xml2json.py ./dataset/VOCdevkit 2012
```

## SSD Training and test (~200 min. at T4 GPU)

```
! python ssd.py
```

## Retinanet Training and test (~200 min. at T4 GPU)

```
! python retinanet.py
```


## Faster R-CNN Training and test (~200 min. at T4 GPU)

```
! python faster_rcnn.py
```

