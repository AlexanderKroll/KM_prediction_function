import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from GNN_functions import *
from metabolite_preprocessing import *
from enzyme_representations import *

import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join

CURRENT_DIR = os.getcwd()

def KM_predicton(substrate_list, enzyme_list):
    #creating input matrices for all substrates:
    print("Step 1/3: Calculating numerical representations for all metabolites.")
    print(".....1(a) Calculating input matrices for Graph Neural Network")
    df_met = metabolite_preprocessing(metabolite_list = substrate_list)
    print(".....1(b) Calculating numerical metabolite representations using a Graph Neural Network")
    df_met = calculate_gnn_representations(df_met)
    #remove temporary metabolite directory:
    shutil.rmtree(join(CURRENT_DIR, "..", "data", "temp_met"))

    print("Step 2/3: Calculating numerical representations for all enzymes.")
    df_enzyme = calcualte_esm1b_vectors(enzyme_list = enzyme_list)

    print("Step 3/3: Making predictions for KM.")
    #Merging the Metabolite and the enzyme DataFrame:
    df_KM = pd.DataFrame(data = {"substrate" : substrate_list, "enzyme" : enzyme_list, "index" : list(range(len(substrate_list)))})
    df_KM = merging_metabolite_and_enzyme_df(df_met, df_enzyme, df_KM)
    df_KM_valid, df_KM_invalid = df_KM.loc[df_KM["complete"]], df_KM.loc[~df_KM["complete"]]
    df_KM_valid.reset_index(inplace = True, drop = True)
    X = calculate_xgb_input_matrix(df = df_KM_valid)
    KMs = predict_KM(X)

    df_KM_valid["KM [mM]"] = KMs
    df_KM = pd.concat([df_KM_valid, df_KM_invalid], ignore_index = True)
    df_KM = df_KM.sort_values(by = ["index"])
    df_KM.drop(columns = ["index"], inplace = True)
    df_KM.reset_index(inplace = True, drop = True)
    
    return(df_KM)




def predict_KM(X):
	bst = pickle.load(open(join(CURRENT_DIR, "..", "data", "saved_models", "xgboost", "xgboost_model_new_KM_esm1b.dat"), "rb"))
	dX = xgb.DMatrix(X)
	KMs = 10**bst.predict(dX)
	return(KMs)


def calculate_xgb_input_matrix(df):
	fingerprints = np.array(list(df["GNN rep"]))
	ESM1b = np.array(list(df["enzyme rep"]))
	X = np.concatenate([fingerprints, ESM1b], axis = 1)
	return(X)



    
def merging_metabolite_and_enzyme_df(df_met, df_enzyme, df_KM):
	df_KM["GNN rep"], df_KM["enzyme rep"] = "", ""
	df_KM["complete"] = True

	for ind in df_KM.index:
		gnn_rep = list(df_met["GNN rep"].loc[df_met["metabolite"] == df_KM["substrate"][ind]])[0]
		esm1b_rep = list(df_enzyme["enzyme rep"].loc[df_enzyme["amino acid sequence"] == df_KM["enzyme"][ind]])[0]

		if gnn_rep == "" or esm1b_rep == "":
			df_KM["complete"][ind] = False
		else:
			df_KM["GNN rep"][ind] = gnn_rep
			df_KM["enzyme rep"][ind] = esm1b_rep
	return(df_KM)







