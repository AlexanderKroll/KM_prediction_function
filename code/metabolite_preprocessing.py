import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import shutil
import pickle
import os
from os.path import join

import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.getcwd()

df_metabolites = pd.read_pickle(join(CURRENT_DIR, "..", "data", "additional_data", "all_substrates.pkl"))

def metabolite_preprocessing(metabolite_list):
	#removing duplicated entries and creating a pandas DataFrame with all metabolites
	df_met = pd.DataFrame(data = {"metabolite" : list(set(metabolite_list))})
	df_met["type"], df_met["ID"] = np.nan, np.nan

	#each metabolite should be either a KEGG ID, InChI string, or a SMILES:
	for ind in df_met.index:
		df_met["ID"][ind] = "metabolite_" + str(ind)
		met = df_met["metabolite"][ind]
		if is_KEGG_ID(met):
			df_met["type"][ind] = "KEGG"
		elif is_InChI(met):
			df_met["type"][ind] = "InChI"
		elif is_SMILES(met):
			df_met["type"][ind] = "SMILES"
		else:
			df_met["type"][ind] = "invalid"
			print(".......Metabolite string '%s' could be neither classified as a valid KEGG ID, InChI string or SMILES string" % met)

	df_met = calculate_atom_and_bond_feature_vectors(df_met)
	N_max = np.max(df_met["number_atoms"].loc[df_met["successfull"]]) + 1
	calculate_input_matrices(df_met = df_met, N_max = N_max)
	return(df_met)

def metabolite_preprocessing_ecfp(metabolite_list):
	#removing duplicated entries and creating a pandas DataFrame with all metabolites
	df_met = pd.DataFrame(data = {"metabolite" : list(set(metabolite_list))})
	df_met["type"], df_met["ID"] = np.nan, np.nan

	#each metabolite should be either a KEGG ID, InChI string, or a SMILES:
	for ind in df_met.index:
		df_met["ID"][ind] = "metabolite_" + str(ind)
		met = df_met["metabolite"][ind]
		if is_KEGG_ID(met):
			df_met["type"][ind] = "KEGG"
		elif is_InChI(met):
			df_met["type"][ind] = "InChI"
		elif is_SMILES(met):
			df_met["type"][ind] = "SMILES"
		else:
			df_met["type"][ind] = "invalid"
			print(".......Metabolite string '%s' could be neither classified as a valid KEGG ID, InChI string or SMILES string" % met)

	df_met = calculate_ecfps(df_met)
	return(df_met)

def maximal_number_of_atoms(df_met):
	N_max = np.max(df_met["number_atoms"].loc[df_met["successfull"]]) + 1
	if N_max > 70:
		print(".......The biggest molecule has over 70 atoms (%s). This will slow down the process of calculating the metabolite representations." % N_max)

def is_KEGG_ID(met):
	#a valid KEGG ID starts with a "C" or "D" followed by a 5 digit number:
	if len(met) == 6 and met[0] in ["C", "D"]:
		try:
			int(met[1:])
			return(True)
		except: 
			pass
	return(False)

def is_SMILES(met):
	m = Chem.MolFromSmiles(met,sanitize=False)
	if m is None:
	  return(False)
	else:
	  try:
	    Chem.SanitizeMol(m)
	  except:
	    print('.......Metabolite string "%s" is in SMILES format but has invalid chemistry' % met)
	    return(False)
	return(True)

def is_InChI(met):
	m = Chem.inchi.MolFromInchi(met,sanitize=False)
	if m is None:
	  return(False)
	else:
	  try:
	    Chem.SanitizeMol(m)
	  except:
	    print('.......Metabolite string "%s" is in InChI format but has invalid chemistry' % met)
	    return(False)
	return(True)


#Create dictionaries for the bond features:
dic_bond_type = {'AROMATIC': np.array([0,0,0,1]), 'DOUBLE': np.array([0,0,1,0]),
                 'SINGLE': np.array([0,1,0,0]), 'TRIPLE': np.array([1,0,0,0])}

dic_conjugated =  {0.0: np.array([0]), 1.0: np.array([1])}

dic_inRing = {0.0: np.array([0]), 1.0: np.array([1])}

dic_stereo = {'STEREOANY': np.array([0,0,0,1]), 'STEREOE': np.array([0,0,1,0]),
              'STEREONONE': np.array([0,1,0,0]), 'STEREOZ': np.array([1,0,0,0])}

##Create dictionaries, so the atom features can be easiliy converted into a numpy array

#all the atomic numbers with a total count of over 200 in the data set are getting their own one-hot-encoded
#vector. All the otheres are lumped to a single vector.
dic_atomic_number = {0.0: np.array([1,0,0,0,0,0,0,0,0,0]), 1.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     3.0: np.array([0,0,0,0,0,0,0,0,0,1]),  4.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     5.0: np.array([0,0,0,0,0,0,0,0,0,1]),  6.0: np.array([0,1,0,0,0,0,0,0,0,0]),
                     7.0:np.array([0,0,1,0,0,0,0,0,0,0]),  8.0: np.array([0,0,0,1,0,0,0,0,0,0]),
                     9.0: np.array([0,0,0,0,1,0,0,0,0,0]), 11.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     12.0: np.array([0,0,0,0,0,0,0,0,0,1]), 13.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     14.0: np.array([0,0,0,0,0,0,0,0,0,1]), 15.0: np.array([0,0,0,0,0,1,0,0,0,0]),
                     16.0: np.array([0,0,0,0,0,0,1,0,0,0]), 17.0: np.array([0,0,0,0,0,0,0,1,0,0]),
                     19.0: np.array([0,0,0,0,0,0,0,0,0,1]), 20.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     23.0: np.array([0,0,0,0,0,0,0,0,0,1]), 24.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     25.0: np.array([0,0,0,0,0,0,0,0,0,1]), 26.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     27.0: np.array([0,0,0,0,0,0,0,0,0,1]), 28.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     29.0: np.array([0,0,0,0,0,0,0,0,0,1]), 30.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     32.0: np.array([0,0,0,0,0,0,0,0,0,1]), 33.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     34.0: np.array([0,0,0,0,0,0,0,0,0,1]), 35.0: np.array([0,0,0,0,0,0,0,0,1,0]),
                     37.0: np.array([0,0,0,0,0,0,0,0,0,1]), 38.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     42.0: np.array([0,0,0,0,0,0,0,0,0,1]), 46.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     47.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     48.0: np.array([0,0,0,0,0,0,0,0,0,1]), 50.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     51.0: np.array([0,0,0,0,0,0,0,0,0,1]), 52.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     53.0: np.array([0,0,0,0,0,0,0,0,0,1]), 54.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     56.0: np.array([0,0,0,0,0,0,0,0,0,1]), 57.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     74.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     78.0: np.array([0,0,0,0,0,0,0,0,0,1]), 79.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     80.0: np.array([0,0,0,0,0,0,0,0,0,1]), 81.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     82.0: np.array([0,0,0,0,0,0,0,0,0,1]), 83.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     86.0: np.array([0,0,0,0,0,0,0,0,0,1]), 88.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     90.0: np.array([0,0,0,0,0,0,0,0,0,1]), 94.0: np.array([0,0,0,0,0,0,0,0,0,1])}

#There are only 5 atoms in the whole data set with 6 bonds and no atoms with 5 bonds. Therefore I lump 4, 5 and 6 bonds
#together
dic_num_bonds = {0.0: np.array([0,0,0,0,1]), 1.0: np.array([0,0,0,1,0]),
                 2.0: np.array([0,0,1,0,0]), 3.0: np.array([0,1,0,0,0]),
                 4.0: np.array([1,0,0,0,0]), 5.0: np.array([1,0,0,0,0]),
                 6.0: np.array([1,0,0,0,0])}

#Almost alle charges are -1,0 or 1. Therefore I use only positiv, negative and neutral as features:
dic_charge = {-4.0: np.array([1,0,0]), -3.0: np.array([1,0,0]),  -2.0: np.array([1,0,0]), -1.0: np.array([1,0,0]),
               0.0: np.array([0,1,0]),  1.0: np.array([0,0,1]),  2.0: np.array([0,0,1]),  3.0: np.array([0,0,1]),
               4.0: np.array([0,0,1]), 5.0: np.array([0,0,1]), 6.0: np.array([0,0,1])}

dic_hybrid = {'S': np.array([0,0,0,0,1]), 'SP': np.array([0,0,0,1,0]), 'SP2': np.array([0,0,1,0,0]),
              'SP3': np.array([0,1,0,0,0]), 'SP3D': np.array([1,0,0,0,0]), 'SP3D2': np.array([1,0,0,0,0]),
              'UNSPECIFIED': np.array([1,0,0,0,0])}

dic_aromatic = {0.0: np.array([0]), 1.0: np.array([1])}

dic_H_bonds = {0.0: np.array([0,0,0,1]), 1.0: np.array([0,0,1,0]), 2.0: np.array([0,1,0,0]),
               3.0: np.array([1,0,0,0]), 4.0: np.array([1,0,0,0]), 5.0: np.array([1,0,0,0]),
               6.0: np.array([1,0,0,0])}

dic_chirality = {'CHI_TETRAHEDRAL_CCW': np.array([1,0,0]), 'CHI_TETRAHEDRAL_CW': np.array([0,1,0]),
                 'CHI_UNSPECIFIED': np.array([0,0,1])}

	
def calculate_ecfps(df_met):
    df_met["successfull"] = True
    df_met["ECFP"] = ""
    df_met["metabolite_similarity_score"] = np.nan
    for ind in df_met.index:
        ID, met_type, met = df_met["ID"][ind], df_met["type"][ind], df_met["metabolite"][ind]
        if met_type == "invalid":
            mol = None
        elif met_type == "KEGG":
        	try:
        		mol = Chem.MolFromMolFile(join(CURRENT_DIR, "..", "data", "mol-files",  met + ".mol"))
        	except:
        		print(".......Mol file for KEGG ID '%s' is not available. Try to enter InChI string or SMILES for the metabolite instead." % met)
        		mol = None
        elif met_type == "InChI":
            mol = Chem.inchi.MolFromInchi(met)
        elif met_type == "SMILES":
            mol = Chem.MolFromSmiles(met)
        if mol is None:
            df_met["successfull"][ind] = False
        else:
        	ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToBitString()
        	df_met["ECFP"][ind] = ecfp
        	df_met["metabolite_similarity_score"][ind] = calculate_metabolite_similarity(df_metabolites = df_metabolites,
                                                                                         mol = mol)
    return(df_met)

def calculate_atom_and_bond_feature_vectors(df_met):
    df_met["successfull"] = True
    df_met["number_atoms"] = 0
    df_met["LogP"], df_met["MW"] = np.nan, np.nan
    df_met["metabolite_similarity_score"] = np.nan
    #Creating a temporary directory to save data for metabolites
    try:
        os.mkdir(join(CURRENT_DIR, "..", "data", "temp_met"))
        os.mkdir(join(CURRENT_DIR, "..", "data", "temp_met", "mol_feature_vectors"))	
    except FileExistsError:
        shutil.rmtree(join(CURRENT_DIR, "..", "data", "temp_met"))
        os.mkdir(join(CURRENT_DIR, "..", "data", "temp_met"))
        os.mkdir(join(CURRENT_DIR, "..", "data", "temp_met", "mol_feature_vectors"))

    for ind in df_met.index:
        ID, met_type, met = df_met["ID"][ind], df_met["type"][ind], df_met["metabolite"][ind]
        if met_type == "invalid":
            mol = None
        elif met_type == "KEGG":
        	try:
        		mol = Chem.MolFromMolFile(join(CURRENT_DIR, "..", "data", "mol-files",  met + ".mol"))
        	except:
        		print(".......Mol file for KEGG ID '%s' is not available. Try to enter InChI string or SMILES for the metabolite instead." % met)
        		mol = None
        elif met_type == "InChI":
            mol = Chem.inchi.MolFromInchi(met)
        elif met_type == "SMILES":
            mol = Chem.MolFromSmiles(met)
        if mol is None:
            df_met["successfull"][ind] = False
        else:
            df_met["number_atoms"][ind] = mol.GetNumAtoms()
            df_met["MW"][ind] = Descriptors.ExactMolWt(mol)
            df_met["LogP"][ind] = Crippen.MolLogP(mol)
            df_met["metabolite_similarity_score"][ind] = calculate_metabolite_similarity(df_metabolites = df_metabolites,
                                                                                         mol = mol)
            calculate_atom_feature_vector_for_mol(mol, ID)
            calculate_bond_feature_vector_for_mol(mol, ID)
    return(df_met)
            
            
def calculate_atom_feature_vector_for_mol(mol, mol_ID):
    #get number of atoms N
    N = mol.GetNumAtoms()
    atom_list = []
    for i in range(N):
        features = []
        atom = mol.GetAtomWithIdx(i)
        features.append(atom.GetAtomicNum()), features.append(atom.GetDegree()), features.append(atom.GetFormalCharge())
        features.append(str(atom.GetHybridization())), features.append(atom.GetIsAromatic()), features.append(atom.GetMass())
        features.append(atom.GetTotalNumHs()), features.append(str(atom.GetChiralTag()))
        atom_list.append(features)
    with open(join(CURRENT_DIR, "..", "data", "temp_met", "mol_feature_vectors",
                    mol_ID + "-atoms.txt"), "wb") as fp:   #Pickling
        pickle.dump(atom_list, fp)
            
def calculate_bond_feature_vector_for_mol(mol, mol_ID):
    N = mol.GetNumBonds()
    bond_list = []
    for i in range(N):
        features = []
        bond = mol.GetBondWithIdx(i)
        features.append(bond.GetBeginAtomIdx()), features.append(bond.GetEndAtomIdx()),
        features.append(str(bond.GetBondType())), features.append(bond.GetIsAromatic()),
        features.append(bond.IsInRing()), features.append(str(bond.GetStereo()))
        bond_list.append(features)
    with open(join(CURRENT_DIR, "..", "data", "temp_met", "mol_feature_vectors",
                    mol_ID + "-bonds.txt"), "wb") as fp:   #Pickling
        pickle.dump(bond_list, fp)



def concatenate_X_and_E(X, E, N, F= 32+10):
    XE = np.zeros((N, N, F))
    for v in range(N):
        x_v = X[v,:]
        for w in range(N):
            XE[v,w, :] = np.concatenate((x_v, E[v,w,:]))
    return(XE)

def calculate_input_matrices(df_met, N_max, save_folder = join(CURRENT_DIR, "..", "data", "temp_met", "GNN_input_data")):

    os.mkdir(save_folder)

    for ind in df_met.index:
        if df_met["successfull"][ind]:
            met_ID = df_met["ID"][ind]
            extras = np.array([df_met["MW"][ind], df_met["LogP"][ind]])
            [XE, X, A] = create_input_data_for_GNN_for_substrates(substrate_ID = met_ID, N_max = N_max, print_error=True)
            if not A is None:
                np.save(join(save_folder, met_ID + '_X.npy'), X) #feature matrix of atoms/nodes
                np.save(join(save_folder, met_ID + '_XE.npy'), XE) #feature matrix of atoms/nodes and bonds/edges
                np.save(join(save_folder, met_ID + '_A.npy'), A)
                np.save(join(save_folder, met_ID + '_extras.npy'), extras)
            else:
                df_met["successfull"][ind] = False


def create_input_data_for_GNN_for_substrates(substrate_ID, N_max, print_error = False):
    try:
        x = create_atom_feature_matrix(mol_name = substrate_ID, N =N_max)
        if not x is None: 
            a,e = create_bond_feature_matrix(mol_name = substrate_ID, N =N_max)
            a = np.reshape(a, (N_max,N_max,1))
            xe = concatenate_X_and_E(x, e, N = N_max)
            return([np.array(xe), np.array(x), np.array(a)])
        else:
            if print_error:
                print(".......Could not create input for substrate ID %s" %substrate_ID)      
            return(None, None, None)
    except:
        return(None, None, None)


def create_bond_feature_matrix(mol_name, N):
    '''create adjacency matrix A and bond feature matrix/tensor E'''
    try:
        with open(join(CURRENT_DIR, "..", "data", "temp_met", "mol_feature_vectors",
                       mol_name + "-bonds.txt"), "rb") as fp:   # Unpickling
            bond_features = pickle.load(fp)
    except FileNotFoundError:
        return(None)
    A = np.zeros((N,N))
    E = np.zeros((N,N,10))
    for i in range(len(bond_features)):
        line = bond_features[i]
        start, end = line[0], line[1]
        A[start, end] = 1 
        A[end, start] = 1
        e_vw = np.concatenate((dic_bond_type[line[2]], dic_conjugated[line[3]],
                               dic_inRing[line[4]], dic_stereo[line[5]]))
        E[start, end, :] = e_vw
        E[end, start, :] = e_vw
    return(A,E)


def create_atom_feature_matrix(mol_name, N):
    try:
        with open(join(CURRENT_DIR, "..", "data", "temp_met", "mol_feature_vectors",
                        mol_name + "-atoms.txt"), "rb") as fp:   # Unpickling
            atom_features = pickle.load(fp)
    except FileNotFoundError:
        return(None)
    X = np.zeros((N,32))
    if len(atom_features) >=N:
        return(None)
    for i in range(len(atom_features)):
        line = atom_features[i]
        x_v = np.concatenate((dic_atomic_number[line[0]], dic_num_bonds[line[1]], dic_charge[line[2]],
                             dic_hybrid[line[3]], dic_aromatic[line[4]], np.array([line[5]/100.]),
                             dic_H_bonds[line[6]], dic_chirality[line[7]]))
        X[i,:] = x_v
    return(X)


###functions for calculating the similarity between metabolites:



def calculate_metabolite_similarity(df_metabolites, mol):
    fp = Chem.RDKFingerprint(mol)
    
    fps = list(df_metabolites["Sim_FP"])
    similarity_vector = np.zeros(len(fps))
    for i in range(len(fps)):
        similarity_vector[i] = DataStructs.FingerprintSimilarity(fp,fps[i])
    return(max(similarity_vector))

