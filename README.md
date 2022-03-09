# Description
This repository contains an easy-to-use python function for the KM prediction model from our paper "Deep learning allows genome-scale prediction of Michaelis constants from structural features". 
Please note that the provided model is not identical to the one presented in the paper: Here, we used enzyme representations that are slightly different. Instead of the UniRep model, here we are using the
ESM-1b model to create the enzyme representations. It was shown that the ESM-1b model outperforms the UniRep model as it is trained with a more up-to-date model for natural language processing 
(with a transformer network instead of a LSTM).

## Predicting Km values for enzyme-substrate pairs
The KM prediction model was only trained with natural enzyme-substrate pairs. Hence, the model will not be good at detecting non-substrates,
but it is only suitable for predicting the KM value if we already know the substrate for an enzyme. Moreover, we only trained our model with 
wild-type ennymes. Therefore, we would not expect that the model to be good at predicting the effect of singe amino acid mutations, as it was
not trained to do so.

## Requirements

- python 3.7
- tensorflow
- jupyter
- pandas
- torch
- numpy
- rdkit
- fair-esm
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

