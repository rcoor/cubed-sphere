import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBList
import os

class DeltaPrepper:
    ''' Generate delta g predictions from datasheets '''

    def __init__(self):
        self.dataframe = pd.DataFrame({'A' : []})

    def load_data(self, input):
        if not input:
            raise ValueError("Input must be defined.")

        # Load curated dataset
        df = pd.read_csv(input, skiprows=21)

        # Convert an array of length 4 to a pandas dataframe
        split_mut = lambda x: pd.Series({'chain':x[0], 'wildtype': x[1], 'position': x[2], 'mutation': x[3] })

        # Split elements in the mutations column into seperate columns
        df_mut = df['Mutations'].apply(lambda m: split_mut(m.split(' ')))
        df = pd.concat([df, df_mut], axis=1)
        self.dataframe = df



''' def download_structure(PDFFileID):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(PDFFileID, pdir="./data/PDB/")
    #structure = MMCIFParser().get_structure(PDFFileID, "PDB/{}.cif".format(PDFFileID))

# Get unique proteins featured in the dataset and download structures
for PDFFileID in df['PDBFileID'].unique():
    download_structure(PDFFileID)

def fetch_structure(PDFFileID):
    # Fetch structure from .cif file
    structure = MMCIFParser().get_structure(PDFFileID, "{}PDB/{}.cif".format(DATA_DIR, PDFFileID))
    # Build structures
    ppb=PPBuilder()
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence()
        print(seq)

for PDFFileID in df['PDBFileID'].unique():
    fetch_structure(PDFFileID)

print(df.head(200))
fetch_structure('1A2P') '''
