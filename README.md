# Drowsiness Detection

Drowsiness detector using `CenterFace` face detector and ResNet18 Drowsiness classifier <br>
<table>
  <tr>
    <th>Awake</th>
    <th>Drowsy</th>
    <th>Sleeping</th>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/Gn48FNx/Whats-App-Image-2022-01-12-at-9-42-53-PM.jpg"  alt="awake" width="300"/>
</td>
    <td><img src="https://i.ibb.co/d5HwYD0/Whats-App-Image-2022-01-12-at-9-43-23-PM.jpg" alt="drowsy" width="300"/>
</td>
    <td><img src="https://i.ibb.co/mvJvph4/Whats-App-Image-2022-01-12-at-9-43-43-PM.jpg" alt="sleeping" width="300"/>
</td>
  </tr>
</table>

## Demo 

[Demo Video](https://drive.google.com/file/d/1RMu-EX5x6tlTGiPn75DY9RO8HhD2UCed/preview)

---
## Index

- [Prerequistes](#prerequistes)
- [About Dataset](#dataset)
- [Clone the repository](#clone-the-repository)
- [Setup python dependencies](#setup-python-dependencies)
- [Running the code](#running-the-code)
- [Citations](#citations)

---
## Prerequistes

This repository has been tested with the following - 

- Ubuntu 20.04
- Python 3.9

---
## Dataset 

### Context

This dataset is just one part of The MRL Eye Dataset, the large-scale dataset of human eye images. It is prepared for classification tasks This dataset contains infrared images in low and high resolution, all captured in various lighting conditions and by different devices. The dataset is suitable for testing several features or trainable classifiers. In order to simplify the comparison of algorithms, the images are divided into several categories, which also makes them suitable for training and testing classifiers.

### Source 

The open sourced dataset is available here "https://www.kaggle.com/kutaykutlu/drowsiness-detection"
The full dataset is available here "http://mrl.cs.vsb.cz/eyedataset"

### Content

In the dataset, we annotated the following properties (the properties are indicated in the following order):

* subject ID; in the dataset, we collected the data of 37 different persons (33 men and 4 women)
* Image ID; the dataset consists of 84,898 images
* gender [0 - man, 1 - woman]; the dataset contains the information about gender for each image (man, woman)
* glasses [0 - no, 1 - yes]; the information if the eye image contains glasses is also provided for each image (with and without the glasses)
* eye state [0 - closed, 1 - open]; this property contains the information about two eye states (open, close)
* reflections [0 - none, 1 - small, 2 - big]; we annotated three reflection states based on the size of reflections (none, small, and big reflections)
* lighting conditions [0 - bad, 1 - good]; each image has two states (bad, good) based on the amount of light during capturing the videos
* sensor ID [01 - RealSense, 02 - IDS, 03 - Aptina]; at this moment, the dataset contains the images captured by three different sensors (Intel RealSense RS 300 sensor with 640 x 480 resolution, IDS Imaging sensor with 1280 x 1024 resolution, and Aptina sensor with 752 x 480 resolution)


## Clone the repository

This is a straight-forward step

```sh
# HTTPS
git clone https://github.com/Aasthaengg/drowsiness-detection.git

# SSH
git clone git@github.com:Aasthaengg/drowsiness-detection.git

cd drowsiness-detection/
```

---

## Setup Python dependencies

The python dependencies are as follows

```json
numpy==1.22.0
opencv-python==4.5.5.62
Pillow==9.0.0
torch==1.10.0+cu113
torchaudio==0.10.0+cu113
torchvision==0.11.1+cu113
typing_extensions==4.0.1
```

You can quickly install them using the following command -
```sh
pip3 install -r requirements.txt
```

---

## Running the code

The application is packaged into `detect.py`.

```sh
python3 detect.py
```

---

## Citations

- https://github.com/Star-Clouds/CenterFace
