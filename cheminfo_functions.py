import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem, Descriptors, Draw, rdFingerprintGenerator

import matplotlib.pyplot as plt
import seaborn as sns

def morgan_fp(df, nBits=1024, radius=3, smilesCol="SMILES"):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    ecfp6 = [gen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in df[smilesCol]]
    ecfp6_name = [f'Bit_{i}' for i in range(nBits)]
    ecfp6_bits = [list(l) for l in ecfp6]
    morgan_df = pd.DataFrame(ecfp6_bits, index=df.index, columns=ecfp6_name)
    return morgan_df

def stacked_morgan_fp(df, nBits=1024, max_radius=8, smilesCol="SMILES"):
    ecfp_bit_cols = [f"Bit_{i}" for i in range(nBits)]
    radius_set = []
    for radius in range(1, max_radius+1):
        ecfp_r = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=radius, nBits=nBits) for smiles in df[smilesCol]]
        ecfp_r_values = [list(l) for l in ecfp_r]
        radius_set.append(ecfp_r_values)
    
    radius_set_arr = np.array(radius_set)
    stacked_fp_arr = np.mean(radius_set_arr, axis=0)

    stacked_morgan_df = pd.DataFrame(stacked_fp_arr, index=df.index, columns=ecfp_bit_cols)

    return stacked_morgan_df


def morgan_fp_multi(df, nBits=1024, radius=3, smilesCol="SMILES"):
    ecfp6 = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(AllChem.GetMorganFingerprintAsBitVect)(Chem.MolFromSmiles(smiles), radius=radius, nBits=nBits) for smiles in df[smilesCol])
    ecfp6_name = [f'Bit_{i}' for i in range(nBits)]
    ecfp6_bits = [list(l) for l in ecfp6]
    morgan_df = pd.DataFrame(ecfp6_bits, index=df.index, columns=ecfp6_name)
    return morgan_df

def describe_smiles_df(smiles_df, smilesCol="SMILES"):
    
    PandasTools.AddMoleculeColumnToFrame(smiles_df, smilesCol=smilesCol, molCol="RDKitMol", includeFingerprints=True)
    PandasTools.AddMurckoToFrame(smiles_df, molCol="RDKitMol", MurckoCol="Murcko_SMILES")
    descriptors = [Descriptors.CalcMolDescriptors(mol) for mol in smiles_df["RDKitMol"]]
    descriptors_df = pd.DataFrame(descriptors)
    expanded_smiles_df = pd.concat([smiles_df, descriptors_df], axis=1)
    return expanded_smiles_df

def describe_smiles_df_multi(smiles_df, smilesCol="SMILES"):

    PandasTools.AddMoleculeColumnToFrame(smiles_df, smilesCol=smilesCol, molCol="RDKitMol", includeFingerprints=True)

    PandasTools.AddMurckoToFrame(smiles_df, molCol="RDKitMol", MurckoCol="Murcko_SMILES")

    descriptors = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(Descriptors.CalcMolDescriptors)(mol) for mol in smiles_df["RDKitMol"])

    descriptors_df = pd.DataFrame(descriptors)

    expanded_smiles_df = pd.concat([smiles_df, descriptors_df], axis=1)

    return expanded_smiles_df


def smiles_desc_profile(descriptors_df, selected_descriptors, smilesCol, murckoSmilesCol, n_sample=3, isCluster=False):
    
    # get sample of instances in dataset
    smile_sample = descriptors_df.sample(n=n_sample, random_state=42)[smilesCol]

    mol_sample = [Chem.MolFromSmiles(smiles) for smiles in smile_sample]

    Draw.MolsToGridImage(mol_sample, molsPerRow=n_sample)

    # get most common murcko structures
    # and their counts
    top5_murcko_smiles = list(descriptors_df[murckoSmilesCol].value_counts()[:n_sample].index)
    top5_murcko_mols = [Chem.MolFromSmiles(smiles) for smiles in top5_murcko_smiles]
    
    top5_murcko_counts = list(descriptors_df[murckoSmilesCol].value_counts()[:n_sample].values)

    Draw.MolsToGridImage(mols=top5_murcko_mols, molsPerRow=n_sample)

    visual_descriptors = descriptors_df[selected_descriptors]

    if not isCluster:
        fig, axes = plt.subplots(1, len(selected_descriptors), figsize=(12,5))
        for descriptor_indx in range(len(selected_descriptors)):
            sns.kdeplot(
                ax=axes[descriptor_indx], 
                data=visual_descriptors[selected_descriptors[descriptor_indx]], 
                fill=True
            )
            if descriptor_indx != 0:
                axes[descriptor_indx].set(
                    ylabel=""
                )

        plt.tight_layout()

    return mol_sample, top5_murcko_mols, top5_murcko_counts

def cluster_smiles_desc_profile(descriptors_df, selected_descriptors, smilesCol, murckoSmilesCol, n_sample=3, clusterCol="Cluster"):
    all_cluster_dict = dict()

    for cluster in set(list(descriptors_df[clusterCol])):
        mol_sample, murcko_top, murcko_top_counts = smiles_desc_profile(
            descriptors_df, 
            selected_descriptors, 
            smilesCol=smilesCol, 
            murckoSmilesCol=murckoSmilesCol,
            n_sample=n_sample,
            isCluster=True
        )
        all_cluster_dict[cluster] = {
            "MolSample": mol_sample,
            "TopMurcko": murcko_top,
            "MurckoCounts": murcko_top_counts
        }
    
    # need to add cluster to hue
    fig, axes = plt.subplots(1, len(selected_descriptors), figsize=(12,5))
    for descriptor_indx in range(len(selected_descriptors)):
        sns.kdeplot(
            ax=axes[descriptor_indx], 
            data=descriptors_df,
            x=selected_descriptors[descriptor_indx],
            fill=True,
            hue=clusterCol
        )
        if descriptor_indx != 0:
            axes[descriptor_indx].set(
                ylabel=""
            )

        plt.tight_layout()

    
    return all_cluster_dict



    

