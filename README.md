# Drowsiness Detection

Drowsiness detector using `CenterFace` face detector and ResNet18 Drowsiness classifier

---
## Index

- [Prerequistes](#prerequistes)
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