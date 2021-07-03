import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

radius = 3
useCounts = True
useFeatures = True


class Ecfp(object):
    def __init__(self):
        self.name = "ecfp"
        self.radius = radius
        self.useCounts = useCounts
        self.useFeatures = useFeatures

    def calc(self, mols):
        fps = [
            AllChem.GetMorganFingerprint(
                mol, self.radius, useCounts=self.useCounts, useFeatures=self.useFeatures
            )
            for mol in mols
        ]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return np.array(nfp, dtype=np.int32)


def get_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fingerprinter = Ecfp()
    return fingerprinter.calc(mols)
