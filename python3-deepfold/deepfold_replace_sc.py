import os
import subprocess
import Bio
import Bio.PDB
from deepfold_predict_ddg import read_ddg_csv

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', Bio.PDB.PDBExceptions.PDBConstructionWarning)

import simtk.openmm.app
import pdbfixer


def create_mutation_file(mutation, pdb_dir, pdb_output_dir):
    pdb_id, mutation, ddg = mutation
    wt, res_index, mutant = mutation[0]

    print("Processing: %s %s-%s-%s" % (pdb_id, wt, res_index, mutant))
    
    pdb_path = os.path.join(pdb_dir, pdb_id+".pdb")

    # Get the PDB sequence
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_path)
    ppb = Bio.PDB.PPBuilder()
    pps = list(ppb.build_peptides(structure))
    pdb_initial_res_index = pps[0][0].get_id()[1]
    assert(len(pps)==1)

    wt_seq = pps[0].get_sequence()

    assert(wt_seq[res_index-pdb_initial_res_index] == wt)

    # Construct the mutant sequence
    mutant_seq = list(wt_seq.lower())
    mutant_seq[res_index-pdb_initial_res_index] = mutant
    mutant_seq = "".join(mutant_seq)

    # Save the output in a file
    output_basename = "%s_%s-%d-%s" % (pdb_id, wt, res_index, mutant)
    sequence_output_path = os.path.join(pdb_output_dir, output_basename+".txt")

    with open(sequence_output_path, "w") as sequence_output_file:
        sequence_output_file.write(mutant_seq)

    # Construct the mutant PDB file
    pdb_output_path = os.path.join(pdb_output_dir, output_basename+".pdb")

    p = subprocess.Popen(["Scwrl4", "-i", pdb_path, "-s", sequence_output_path, "-o", pdb_output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if p.returncode != 0:
        print("Error running Scwrl4 on:", pdb_output_path)
        print(err)

    with open(os.path.join(pdb_output_dir, output_basename+".scwrl"), "w") as scwrl_output_file:
        scwrl_output_file.write(out)

    # Fix the output structure
    fixer = pdbfixer.PDBFixer(filename=pdb_output_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(pdb_output_path, "w") as pdb_output_file:
        simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, pdb_output_file, keepIds=True)



if __name__ == '__main__':
    import joblib

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--pdb-dir",
                        help="Location of PDB files")
    parser.add_argument("--pdb-output-dir",
                        help="Location for the new PDB files")
    parser.add_argument("--ddg-csv-filename", dest="ddg_csv_filename",
                        help="CSV file containing ddG data")
    parser.add_argument("--n-proc", metavar="VAL", type=int, default=1,
                        help="Number of processes (default: %(default)s)")
    args = parser.parse_args()
    
    mutations = read_ddg_csv(args.ddg_csv_filename)

    # Create the output dir
    if not os.path.exists(args.pdb_output_dir):
        os.mkdir(args.pdb_output_dir)

    # Copy the PDB files to the output dir
    filtered_mutations = []

    for mutation in mutations:
        pdb_id, _, _ = mutation
        
        # Check if PDB file exists
        pdb_path = os.path.join(args.pdb_dir, pdb_id+".pdb")

        if not os.path.exists(pdb_path):
            print("File does not exists: %s" % pdb_path)
            continue

        # Make a copy of the PDB file
        if not os.path.exists(os.path.join(args.pdb_output_dir, pdb_id+".pdb")):
            subprocess.call(["cp", pdb_path, args.pdb_output_dir])

        filtered_mutations.append(mutation)

    # Run PDB files for each mutation
    joblib.Parallel(n_jobs=args.n_proc, batch_size=1)(joblib.delayed(create_mutation_file)(mutation, args.pdb_dir, args.pdb_output_dir) for mutation in filtered_mutations)
