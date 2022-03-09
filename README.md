# Description
This repository contains any easy-to-use python function for the KM prediction model from our paper "Deep learning allows genome-scale prediction of Michaelis constants from structural features". 
Please note that the provided model is not identical to the one presented in the paper: The used enzyme representations are slightly different. Instead of the UniRep model, here we are using the
ESM-1b model to create the enzyme representations. It was shown that the ESM-1b model outperforms the UniRep model as it is trained with a more up-to-date model for natural language processing 
(with a transformer network instead of a LSTM).

## Requirements

- python 3.7
- tesnsorlow
- jupyter
- pandas
- scikit-learn
- rdkit
- zeep
- matplotlib
- py-xgboost

The listed packaged can be installed using conda and anaconda:

```bash
pip install torch
pip install numpy
pip install xgboost
pip install tensorflow
pip install fair-esm
conda install -c rdkit rdkit
```

## Content

There exist a Jupyter notebook "Tutorial KM prediction.ipynb" in the folder "code" that contains an example on how to use the KM prediction function.

## Problems/Questions
If you face any issues or problems, please open an issue.

