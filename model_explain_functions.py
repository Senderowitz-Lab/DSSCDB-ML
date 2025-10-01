from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem, Descriptors, Draw, rdFingerprintGenerator, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput
from collections import Counter
import re
from io import BytesIO 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import gc, os
import pandas as pd 
import numpy as np
import shap
import joblib
from rdkit.Chem.MolStandardize import rdMolStandardize


def _prepareMol(mol, kekulize):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc

def moltopng(mol, molSize=(900,400), kekulize=True, drawer=None, **kwargs):
    mc = _prepareMol(mol, kekulize)

    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])

    drawer.DrawMolecule(mc, **kwargs)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    return BytesIO(png)

def includeRingMembership(s, n):
    r=';R]'
    d="]"
    return r.join([d.join(s.split(d)[:n]), d.join(s.split(d)[n:])])

def includeDegree(s, n, d):
    r=';D'+str(d)+']'
    d="]"
    return r.join([d.join(s.split(d)[:n]), d.join(s.split(d)[n:])])

def writePropsToSmiles(mol, smi, order):
    finalsmi = smi
    for i,a in enumerate(order):
        atom = mol.GetAtomWithIdx(a)
        if atom.IsInRing():
            finalsmi = includeRingMembership(finalsmi, i+1)
        finalsmi = includeDegree(finalsmi, i+1, atom.GetDegree())
    return finalsmi

def getSubstructureSmi(mol, atomID, radius):
    """
    Return (smi, smi_with_props) for the radius=radius environment around atomID.
    Safe for multi-fragment molecules (only uses rootedAtAtom on single-fragment mols).
    """
    if mol is None:
        return "", ""

    if radius > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)  # list of bond indices
        atoms = {atomID}
        for bidx in env:
            b = mol.GetBondWithIdx(bidx)
            atoms.add(b.GetBeginAtomIdx())
            atoms.add(b.GetEndAtomIdx())
        atomsToUse = sorted(atoms)
    else:
        env = []                      # explicit empty list is fine
        atomsToUse = [atomID]

    # only root if the parent mol is a single fragment
    single_frag = (len(Chem.GetMolFrags(mol, asMols=False)) == 1)

    kwargs = dict(
            atomsToUse=atomsToUse,
            bondsToUse=env,
            allHsExplicit=True,
            allBondsExplicit=True,
        )
    if single_frag:
        kwargs["rootedAtAtom"] = atomID

    smi = Chem.MolFragmentToSmiles(mol, **kwargs)

    # optional: preserve your annotated/propped SMILES
    # if the property is missing, skip the rewrite
    try:
        order = mol.GetProp("_smilesAtomOutputOrder")
        # avoid eval on arbitrary strings; RDKit stores this as a comma-separated list in many workflows
        if isinstance(order, str):
            # try to parse either "1,2,3" or "[1,2,3]"
            if order.strip().startswith("["):
                order = eval(order)  # if you control this value; otherwise parse safely
            else:
                order = [int(x) for x in order.split(",") if x.strip().isdigit()]
        smi2 = writePropsToSmiles(mol, smi, order)
    except Exception:
        smi2 = smi

    return smi, smi2

def getSubstructureDepiction(mol, atomID, radius, molSize=(450,200)):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    return moltopng(mol, molSize=molSize, highlightAtoms=atomsToUse, highlightAtomColors={atomID:(0.3, 0.3, 1)})

def depictBit_new(bitId, mol, molSize=(450, 200), RADIUS=3, N_BITS=1024):
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    ao = AdditionalOutput(); ao.AllocateBitInfoMap()
    _ = gen.GetFingerprint(mol, additionalOutput=ao)
    infomap = ao.GetBitInfoMap()
    if bitId not in infomap or not infomap[bitId]:
        # fallback: depict the atom with highest degree (arbitrary but safe)
        aid, rad = (0, 0)
    else:
        aid, rad = infomap[bitId][0]
    return getSubstructureDepiction(mol, aid, rad, molSize=molSize)


def retrieveSHAPBits_MolBI(ylabels, X_test, smiles_df, smiles_col='SMILES', RADIUS=3, N_BITS = 1024):
    #make ylabels be a list of str
    shap_bits = []
    for label in ylabels:
        if 'Bit' in label:
            shap_bits.append(int(re.findall(r'\d+', label)[0]))

    X_test_smiles = smiles_df[smiles_col].loc[X_test.index]
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

    shap_bit_mol_bi = {}
    for shap_bit in shap_bits:
        X_test_smiles_bit_true = X_test_smiles.loc[X_test[f'Bit_{shap_bit}']==1]

        mol_bi = {}
        for smiles in X_test_smiles_bit_true:
            ao = AdditionalOutput()
            ao.AllocateBitInfoMap()
            _ = gen.GetFingerprint(Chem.MolFromSmiles(smiles), additionalOutput= ao)
            infomap = ao.GetBitInfoMap()
            #keep only if this bit has at least one center in this molecule
            if shap_bit in infomap and infomap[shap_bit]:
                mol_bi[smiles] = infomap
        #store non-empty maps
        if mol_bi:
            shap_bit_mol_bi[shap_bit] = mol_bi

    return shap_bit_mol_bi


def resolveSHAPBits(shap_bit_mol_bi):
    shapBit_smarts = {}
    shapBit_smiles = {}
    shapBit_smarts_maj = {}
    shapBit_smiles_maj = {}
    for shap_bit in shap_bit_mol_bi:
        mol_bi_dict = shap_bit_mol_bi[shap_bit]

        smarts_list = []
        smiles_list = []
        for smile in mol_bi_dict:
            aid, rad = mol_bi_dict[smile][shap_bit][0]
            smi, smart = getSubstructureSmi(Chem.MolFromSmiles(smile), aid, rad)
            smarts_list.append(smart.strip())
            smiles_list.append(smi.strip())

        shapBit_smarts[shap_bit] = Counter(smarts_list)
        shapBit_smiles[shap_bit] = Counter(smiles_list)
        shapBit_smarts_maj[shap_bit] = shapBit_smarts[shap_bit].most_common()[0]
        shapBit_smiles_maj[shap_bit] = shapBit_smiles[shap_bit].most_common()[0]


    return shapBit_smarts, shapBit_smarts_maj, shapBit_smiles_maj


def shapBit_image(shapBit_smarts_maj, molSize=(450,200), **kwargs):
    bitNum_image = {}
    for shapBit in shapBit_smarts_maj:
        smarts = shapBit_smarts_maj[shapBit][0]
        mol = Chem.MolFromSmarts(smarts)
        if len(mol.GetAtoms()) < 3:
            for a in mol.GetAtoms():
                a.SetProp('atomLabel', smarts)
      
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])

        Options = drawer.drawOptions()
        Options.explicitMethyl = True
        Options.addStereoAnnotation = True
        Options.prepareMolsBeforeDrawing = False
        Options.includeMetadata = True
        drawer.SetDrawOptions(Options)

        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()

        bitNum_image[shapBit] = BytesIO(png)

    return bitNum_image

def shap_mol_fp(explanation, X_test, P_test, title, smiles_df, left=-500, right=770, **kwargs):

    ax = shap.plots.beeswarm(explanation, show=False, color_bar=False, group_remaining_features=False,
                             )
    fig = ax.figure
    fig.set_figheight(22)
    fig.set_figwidth(30)

    ylabels = ax.get_yticklabels()
    ylabels_text = [label.get_text() for label in ylabels]

    # robust: works whether P_test is a Series or a DataFrame
    vals = np.asarray(P_test).astype(float).ravel()
    range_min = float(np.nanmin(vals))
    range_max = float(np.nanmax(vals))
    fig.suptitle(f"{title}: [{round(range_min, 2)} to {round(range_max, 2)}]", 
                    fontsize=30,
                    fontweight="bold",
                    x=0.5,
                    ha='center'
        )

    shap_bit_mol_bi = retrieveSHAPBits_MolBI(ylabels_text, X_test, smiles_df)
    shapBit_smarts, shapBit_smarts_maj, shapBit_smiles_maj = resolveSHAPBits(shap_bit_mol_bi)
    shapBit_image_dict = shapBit_image(shapBit_smiles_maj)

    # now have each important bit and their image
    for shapBit in shapBit_image_dict:
        im = plt.imread(shapBit_image_dict[shapBit])
        ib = OffsetImage(im, zoom=0.55)
        ib.image.axes = ax
        shapBit_label = f"Bit_{shapBit}"
        shapBit_idx = ylabels_text.index(shapBit_label)

        ab = AnnotationBbox(
            ib,
            ylabels[shapBit_idx].get_position(),
            xybox=(left, 0.),
            xycoords='data',
            boxcoords='offset points',
            frameon=False
        )
        ax.add_artist(ab)

    i = 0
    used_mols = []
    for shapBit in shap_bit_mol_bi:
        # skip bits with no consensus pattern
        if shapBit not in shapBit_smarts_maj:
            continue
        smarts = shapBit_smarts_maj[shapBit][0]
        molCandidates = []
        mws = []
        for smiles in shap_bit_mol_bi[shapBit]:
            mol = Chem.MolFromSmiles(smiles)
            smart_query = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(smart_query):
                canonSmiles_mol = Chem.MolToSmiles(mol)
                if canonSmiles_mol in used_mols:
                    continue
                molCandidates.append(mol)
                mws.append(Descriptors.HeavyAtomMolWt(mol))
        if not molCandidates:
            try:
                first_smiles = next(iter(shap_bit_mol_bi[shapBit].keys()))
            except StopIteration:
                continue
            lowMW_mol = Chem.MolFromSmiles(first_smiles)
            if lowMW_mol is None:
                continue
        else:
            lowMw_mol = molCandidates[pd.Series(mws).idxmin()]
#            matches = mol.GetSubstructMatches(smart_query)
#            canonSmiles_mol = Chem.MolToSmiles(mol)
#            if (len(matches) !=0) and (canonSmiles_mol not in used_mols):
#                molCandidates.append(mol)
#                mws.append(Descriptors.HeavyAtomMolWt(mol))

#        lowMw_mol = molCandidates[pd.Series(mws).idxmin()]
        used_mols.append(Chem.MolToSmiles(lowMw_mol))

        im = plt.imread(depictBit_new(bitId=shapBit, mol=lowMw_mol, molSize=(600, 250)))
        ib = OffsetImage(im, zoom=1.0)
        ib.image.axes = ax

        shapBit_label = f"Bit_{shapBit}"
        try:
            shapBit_idx = ylabels_text.index(shapBit_label)
        except ValueError:
            continue

        if i%2 ==0:
            offset = 0
        else:
            offset = 500

        ab = AnnotationBbox(
            ib,
            ylabels[shapBit_idx].get_position(),
            xybox=(right+ offset, 0),
            xycoords='data',
            boxcoords='offset points',
            frameon=False
        )
        ax.add_artist(ab)

        i = i +1 

    ax.set_xticklabels(
        labels = ax.xaxis.get_ticklabels(),
        fontsize = 24
    )

    ax.set_yticklabels(
        labels = ax.yaxis.get_ticklabels(),
        fontsize = 24
    )

    ax.set_xlabel(
        ax.xaxis.get_label().get_text(),
        fontsize = 26,
        labelpad = 10
    )

    plt.show()

def get_Explanation_object(path_to_model, unimputed_descriptors, target, splits_tuple,
                           ):

    # 1) Load model via memmap (no copy)
    pipe = joblib.load(path_to_model, mmap_mode="r")
    pre  = pipe.named_steps["columntransformer"]
    est  = pipe.named_steps["randomforestregressor"]

    # 2) Split
    X_train = unimputed_descriptors.loc[splits_tuple[0]]
    X_test = unimputed_descriptors.loc[splits_tuple[1]]
    y_train = target.loc[splits_tuple[0]]
    y_test = target.loc[splits_tuple[1]]

    # 3) Transform TRAIN only 
    X_train_t = pre.transform(X_train)

    # 4) KMeans background (same k=100)
    X_train_summary = shap.kmeans(X_train_t, 100)
    X_train_summary_df = pd.DataFrame(X_train_summary.data, columns=X_train.columns)

    # 5) Free big train arrays before touching TEST
    del X_train_t, X_train, y_train
    gc.collect()

    # transform X_test
    X_test_t = pre.transform(X_test)
    feat_names = pre.get_feature_names_out()
    X_test_t_df  = pd.DataFrame(X_test_t,  index=X_test.index,  columns=feat_names)

    # 7) Build explainer
    explainer = shap.Explainer(est, X_train_summary_df)

    # 8) Predict using transformed test (avoid re-transforming)
    P_eval = est.predict(X_test_t)
    P_eval_df = pd.DataFrame({"prediction": P_eval}, index=X_test.index)

    # 9) Explain 
    explanation = explainer(X_test_t_df,check_additivity=False)

    return explanation, X_test_t_df, P_eval_df