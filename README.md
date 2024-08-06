# Ekstra Bladet new reranker 

This repository contains the code and the pre-trained models for the DArgk team submission to the ACM RecSys Challenge 2024. 


## Proprocessing
The notebook `01-DataPreprocess.ipynb` is intended for preprocessing the `articles`, `behaviors` and `behaviors` part of the dataset. 

The notebook `02-Img-embs.ipynb` encodes the Resnet embedding of the images into a more 128-dimension vector using an autoencoder.

## Generating the predictions

This repository contains all the required files, but the dataset, for generating the rankings as submitted to the challenge. Steps for reproducing the results:

1. Recreate the conda environment using the `environment.yml` file.
```bash 
conda env create -f environment.yml
```
2. Place the dataset files as provided in the folder dataset.
3. Generate the predictions.
```bash
python inferv1_img_bce_val.py
```
4. To format the output of the predictions.
```bash
python to_zip_format.py --exp v1_img_bce_val_epoch_9
```

## Training the model

The model was trained for 8 epochs in the training dataset and finetuned for 1 each on validation.

For executing one epoch of training in the training dataset: 

```bash
python train_1_img_bce_epoch.py
```

For fine-tuning one epoch using the validation dataset: 

```bash
python train_1_img_bce_val_epoch.py
```

## Contact info:

* [Antonela Tommasel](https://tommantonela.github.io) (antonela.tommasel@isistan.unicen.edu.ar)
* [Juan Manuel Rodriguez](https://sites.google.com/site/rodriguezjuanmanuel/home) (jmro@cs.aau.dk)
