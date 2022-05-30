# Description
This repository contains an easy-to-use python function for the KM prediction model from our paper "Deep learning allows genome-scale prediction of Michaelis constants from structural features". 
Please note that the provided model is not identical to the one presented in the paper: Here, we used enzyme representations that are slightly different. Instead of the UniRep model, here we are using the
ESM-1b model to create the enzyme representations. It was shown that the ESM-1b model outperforms the UniRep model as it is trained with a more up-to-date model for natural language processing (with a transformer network instead of a LSTM).

## Predicting Km values for enzyme-substrate pairs
The KM prediction model was only trained with natural enzyme-substrate pairs. Hence, the model will not be good at detecting non-substrates,
but it is only suitable for predicting the KM value if we already know the substrate for an enzyme. Moreover, we only trained our model with 
wild-type ennymes. Therefore, we would not expect that the model to be good at predicting the effect of singe amino acid mutations, as it was
not trained to do so.

## Using KEGG Compound IDs as substrate representations
If you wish to use KEGG Compound IDs as inputs for the substrates, you need to unzip a zipped file called "mol-files", which is in the folder "data". The unzipped folder "mol-files" has to be stored in the folder "data".

Alternatively, you can use InChI strings and SMILES strings as substrate representations.

## Requirements

- python 3.7
- tensorflow 2.3.1
- jupyter
- pandas 1.1.3
- torch 1.7.1
- numpy 
- rdkit 2020.09.1
- fair-esm 0.3.1
- py-xgboost 1.3.1

The listed packaged can be installed using conda and anaconda:

```bash
pip install torch
pip install numpy
pip install tensorflow
pip install fair-esm
conda install -c conda-forge py-xgboost=1.3.3
conda install -c rdkit rdkit
```

## Content

There exist a Jupyter notebook "Tutorial KM prediction.ipynb" in the folder "code" that contains an example on how to use the KM prediction function.

## Problems/Questions
If you face any issues or problems, please open an issue.

