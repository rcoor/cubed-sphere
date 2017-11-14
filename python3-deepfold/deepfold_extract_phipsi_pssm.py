import sys
import os
import glob
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit
import Bio.PDB.Polypeptide
import numpy as np

import deepfold_grid as grid

# DSSP to SS-index(H, E, C)
# H: HGI
# E: EB
# C: STC
def dssp2i(ss):
    '''Convert 8-class DSSP score into three numeric classes'''
    if ss in ("H", "G", "I"):
        return 0
    elif ss in ("E", "B"):
        return 1
    elif ss in ("S", "T", "C", " "):
        return 2
    else:
        # Used for unobserved SS
        return 3

def parse_pssm(pssm_filename):

    pssm_file = open(pssm_filename)

    items = 0
    prev_line = None
    aas = []
    aas2 = []
    sequence_profile = []
    aa_freqs = []
    sequence = ""
    for line in pssm_file:

        line = line.strip()

        if len(line) == 0:
            continue
        
        split_line = line.split()

        # Detect number of items in first iteration
        if items==0 and line[0] == "1":
            items = len(split_line)
            
            header = prev_line
            aas = header.split()[:20]
            aas2 = header.split()[20:40]

        prev_line = line
            
        if items==0 or len(split_line) != items:
            continue

        # extract residue number and name
        res_index, res_name = split_line[:2]
        sequence += res_name
        
        # remove first and last two entries
        split_line = split_line[2:-2]

        # Check that there are 40 values
        assert(len(split_line) == 40)

        # Use the first 20
        scores = split_line[:20]

        sequence_profile.append(np.zeros(20))
        for aa, score in zip(aas, scores):
            standard_index = Bio.PDB.Polypeptide.one_to_index(aa)
            sequence_profile[-1][standard_index] = score

        # Use the last 20
        freqs = split_line[20:]
        aa_freqs.append(np.zeros(20))
        for aa, freq in zip(aas2, freqs):
            standard_index = Bio.PDB.Polypeptide.one_to_index(aa)
            aa_freqs[-1][standard_index] = freq

    sequence_profile = np.array(sequence_profile)
    aa_freqs = np.array(aa_freqs)/100.

    # Apply sigmoid
    sequence_profile = 1.0/(1+np.exp(-sequence_profile))

    pssm_file.close()
    
    return sequence_profile, sequence, aa_freqs
    

# def parse_alignment(pdb_id, pdb_sequence, alignment_filename):

#     lines = open(alignment_filename).readlines()

#     # Jump to last results block
#     start_line = 0
#     for i,line in enumerate(lines):
#         if "Results from round" in line:
#             start_line = i

#     main_block = False
#     first_main_block = False
#     sequences = []
#     keys = []
#     seq_counter = 0
#     for i in range(start_line, len(lines)):

#         line = lines[i].strip()

#         if line.startswith(pdb_id):
#             if not main_block:
#                 first_main_block = True
#             main_block = True
#             seq_counter = 0

#         if "Database" in line:
#             main_block = False
            
#         if not main_block:
#             continue
            
#         if line == "":
#             seq_counter = 0
#             first_main_block = False
#             continue

#         if first_main_block:
#             print "*** ", line
#         else:
#             print line
#         split_line = line.split()
#         if len(split_line) == 4:
#             seq = split_line[2]
#             key = split_line[0]
#         elif len(split_line) == 2:
#             seq = split_line[1]
#             key = split_line[0]

#         if first_main_block:
#             sequences.append([])
#             keys.append(key)
#             sequences[seq_counter] = ""
#         sequences[seq_counter] += seq
#         # if key not in sequences:
#         #     sequences[key] = ""

#         # sequences[key] += seq
#         print seq
#         seq_counter += 1

#     pdb_index = 0
#     for i in range(1, len(sequences)):
#         assert(len(sequences[pdb_index]) == len(sequences[i]))
        
#     print "Alignment0"
#     print "%8s %s" % (pdb_id, sequences[pdb_index])
#     for i in range(1, len(sequences)):
#         print "%8s %s" % (keys[i],sequences[i])

#     for i in range(len(sequences[pdb_index]))[::-1]:
#         aa = sequences[pdb_index][i]

#         if aa == '-':
#             for s in range(len(sequences)):
#                 sequences[s] = sequences[s][:i] + sequences[s][i+1:]

#     print "Alignment"
#     print "%8s %s" % (pdb_id, sequences[pdb_index])
#     for i in range(1, len(sequences)):
#         print "%8s %s" % (keys[i],sequences[i])

#     print sequences[pdb_index]
#     print pdb_sequence
#     assert(sequences[pdb_index] == str(pdb_sequence))

#     counts = np.zeros([len(pdb_sequence), 20])
#     for i in range(len(sequences[pdb_index])):

#         aas = []
#         for s in range(1, len(sequences)):
#             print keys[s], i, len(sequences[s])
#             aa = sequences[s][i]
#             if aa != '-':
#                 try:
#                     aa_index = Bio.PDB.Polypeptide.one_to_index(aa)
#                     aas.append(aa_index)                        
#                 except KeyError:
#                     pass
#         new_counts = np.bincount(aas, minlength=20)
#         counts[i] = new_counts/float(sum(new_counts))
#     return counts

    
def extract_aa_info(pdb_filename, pssm_filename, alignment_filename, dssp_executable):

    pdb_id = os.path.basename(pdb_filename).split('.')[0]

    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_filename)
    ppb = Bio.PDB.PPBuilder()
    pps = ppb.build_peptides(structure)
    CAs = [atom for atom in structure[0].get_atoms() if atom.id == 'CA']
    assert(len(pps) == 1 and len(pps[0])==len(CAs))
    pdb_sequence = pps[0].get_sequence()

    phipsi = np.array(pps[0].get_phi_psi_list())
    
    # Calculate secondary structure
    try:
        dssp = Bio.PDB.DSSP(model=structure[0], pdb_file=pdb_filename, dssp=dssp_executable)
    except:
        dssp = Bio.PDB.DSSP(model=structure[0], in_file=pdb_filename, dssp=dssp_executable)
    ss = []
    for res in structure[0].get_list()[0]:
        try:
            ss.append(dssp2i(res.xtra["SS_DSSP"]))
        except:
            ss.append(3)
    ss = np.array(ss, dtype=np.int8)
    # ss = np.array([dssp2i(res.xtra["SS_DSSP"]) for res in structure[0].get_list()[0]], dtype=np.int8)
    ss_one_hot = np.zeros((len(ss), 4))
    ss_one_hot[np.arange(len(ss)), ss] = 1

    ss_one_hot_default = np.zeros(4)
    ss_one_hot_default[-1] = 1
    
    # Read in PDB file
    pdb = simtk.openmm.app.PDBFile(pdb_filename)

    # # Extract positions
    positions = pdb.getPositions()
    # CA_positions = np.array([atom.coord for atom in pps[0].get_ca_list()])

    # Parse position-specif scoring matrix
    pssm, sequence, aa_freqs = parse_pssm(pssm_filename)

    # parse_alignment(pdb_id, pdb_sequence, alignment_filename)
   
    # Add column for unobserved values
    pssm = np.hstack((pssm, np.zeros((pssm.shape[0], 1))))

    # Define PSSM default
    pssm_default = np.zeros(21)
    pssm_default[-1] = 1

    aa_freqs = np.hstack((aa_freqs, np.zeros((aa_freqs.shape[0], 1))))
    
    # Create structured array for positions
    position_features = np.empty(shape=(len(positions), 1), dtype=[('name','a5'),
                                                                    ('res_index', int),
                                                                    ('x',np.float32), ('y',np.float32), ('z',np.float32)])
    # Iterate over chain,residue,atoms and extract features
    for chain in pdb.getTopology().chains():
        for residue in chain.residues():
            # print residue.id, residue.name
            for atom in residue.atoms():
                index = atom.index
                position = list(positions[index].value_in_unit(simtk.unit.angstrom))
                position_features[index] = tuple([atom.name, residue.index]+position)

    
    assert(str(sequence) == str(pdb_sequence))

    # Create one-hot encoding of amino acids
    aa_indices = [Bio.PDB.Polypeptide.one_to_index(aa) for aa in sequence]
    aa_one_hot = np.zeros((len(aa_indices), 20))
    aa_one_hot[np.arange(len(aa_indices)), aa_indices] = 1

    # Add column for unobserved values
    aa_one_hot_w_unobserved = np.hstack((aa_one_hot, np.zeros((aa_one_hot.shape[0], 1))))

    # Define aa_one_hot default value
    aa_one_hot_w_unobserved_default = np.zeros(21)
    aa_one_hot_w_unobserved_default[-1] = 1 

    return pdb_id, position_features, np.array(aa_indices), aa_one_hot, aa_one_hot_w_unobserved, aa_one_hot_w_unobserved_default, pssm, pssm_default, ss, ss_one_hot, ss_one_hot_default, aa_freqs, phipsi

        

if __name__ == '__main__':

    from argparse import ArgumentParser
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = ArgumentParser()
    parser.add_argument("--coordinate-system", choices=[e.name for e in grid.CoordinateSystem], default=grid.CoordinateSystem.spherical.name,
                        help="Which coordinate system to use (default: %(default)s)")
    parser.add_argument("--z-direction", choices=[e.name for e in grid.ZDirection], default=grid.ZDirection.outward.name,
                        help="Which direction to choose for z-axis (default: %(default)s)")
    parser.add_argument("--pdb-input-dir", dest="pdb_input_dir",
                        help="Location of pdbs")
    parser.add_argument("--seq-profile-input-dir", dest="seq_profile_input_dir",
                        help="Location of pssm files")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Where to dump features")
    parser.add_argument("--dssp-executable", dest="dssp_executable",
                        help="Location of DSSP executable")
    parser.add_argument("--max-radius", dest="max_radius", default=12.0,
                        help="Maximum radius used in grid")
    parser.add_argument("--bins-per-angstrom", dest="bins_per_angstrom", default=0.5,
                        help="How many bins to use per angstromg (in r-dimension and for maximum r)")
    parser.add_argument("--include-seq-profile", dest="include_seq_profile",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to include PSSM feature")

    options = parser.parse_args()

    coordinate_system = grid.CoordinateSystem[options.coordinate_system]
    z_direction = grid.ZDirection[options.z_direction]
    
    n_features = 21+1
    residue_features = ["aa_one_hot_w_unobserved", "residue_index"]
    if options.include_seq_profile:
        n_features += 21
        residue_features = ["aa_one_hot_w_unobserved", "pssm", "residue_index"]
    
    pdb_filenames = glob.glob(os.path.join(options.pdb_input_dir, "*.pdb"))    

    output_array = []
    for pdb_filename in pdb_filenames:

        print(pdb_filename)
        
        pdb_id = os.path.basename(pdb_filename).split('.')[0]
        pssm_input_file = os.path.join(options.seq_profile_input_dir, pdb_id + ".pssm")
        alignment_input_file = os.path.join(options.seq_profile_input_dir, pdb_id + ".aln")

        # try:
        pdb_id, position_features, aa, aa_one_hot, aa_one_hot_w_unobserved, aa_one_hot_w_unobserved_default, pssm, pssm_default, ss, ss_one_hot, ss_one_hot_default, aa_freqs, phipsi = extract_aa_info(pdb_filename, pssm_input_file, alignment_input_file, options.dssp_executable)

        if len(aa) > 700:
            continue

        output = np.zeros([700,21+21+21+4+2])

        output[:,0:21] = aa_one_hot_w_unobserved_default
        output[:len(aa_one_hot),0:21] = aa_one_hot_w_unobserved
        
        output[:,21:42] = pssm_default
        output[:len(aa_one_hot),21:42] = pssm
        
        output[:,42:63] = pssm_default
        output[:len(aa_one_hot),42:63] = aa_freqs

        output[:,63:67] = ss_one_hot_default
        output[:len(aa_one_hot),63:67] = ss_one_hot

        output[:,67:69] = 0.0
        output[:len(aa_one_hot),67:69] = phipsi

        output_array.append(output)
        # except KeyError:        # DSSP raises a KeyError in rare cases (perhaps non-standard AA?)
        #     continue
        # except AssertionError as e:
        #     print >> sys.stderr, "AssertionError: ", e, "Skipping"

        # Save protein level features
        # if not os.path.exists(options.output_dir):
        #     os.makedirs(options.output_dir)
        # np.savez_compressed(os.path.join(options.output_dir, "%s_protein_features"%pdb_id),
        #                     ss = ss,
        #                     ss_one_hot = ss_one_hot,
        #                     aa_one_hot = aa_one_hot,
        #                     aa_one_hot_w_unobserved = aa_one_hot_w_unobserved,
        #                     aa_one_hot_w_unobserved_default = aa_one_hot_w_unobserved_default,
        #                     pssm = pssm,
        #                     pssm_default = pssm_default,
        #                     aa_freqs = aa_freqs,
        #                     residue_features=residue_features,
        #                     coordinate_system = np.array(coordinate_system.value, dtype=np.int32),
        #                     z_direction = np.array(z_direction.value, dtype=np.int32),
        #                     max_radius = np.array(options.max_radius, dtype=np.float32), # angstrom
        #                     n_features = np.array(n_features, dtype=np.int32),
        #                     bins_per_angstrom = np.array(options.bins_per_angstrom, dtype=np.float32),
        #                     n_residues=np.array(len(np.unique(position_features[['res_index']].view(int))), dtype=np.int32))

        # # np.savez_compressed(os.path.join(output_dir, "%s_aa"%pdb_id),
        # #                     aa_one_hot = aa_one_hot)
        # # np.savez_compressed(os.path.join(output_dir, "%s_ss"%pdb_id),
        # #                     ss=ss)
        
        # embed_in_grid(pdb_id, options.output_dir, position_features, 
        #               max_radius=options.max_radius,
        #               n_features = n_features,
        #               bins_per_angstrom = options.bins_per_angstrom,
        #               coordinate_system = coordinate_system,
        #               z_direction = z_direction)
        
        # sys.exit()
        # print output
    output_array = np.stack(output_array)
    np.savez_compressed(os.path.join(options.output_dir, "phipsi_features"),
                        features=output_array)
