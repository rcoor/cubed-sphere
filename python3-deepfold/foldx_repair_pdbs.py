import sys
import glob

def foldx_repair_pdb(pdb_filename, output_dir, foldx_executable):

    rotabase_locations = os.path.join(os.path.basename(foldx_executable), 'rotabase.txt')
    
    command_line = "%s -c RepairPDB --pdb=%s --rotabaseLocation=%s > /dev/null" % (foldx_executable, pdb_filename, rotabase_location)



if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--pdb-input-dir", dest="pdb_input_dir",
                        help="Location of input PDB files ")
    parser.add_argument("--pdb-output-dir", dest="pdb_output_dir",
                        help="Where to dump output PDB files ")
    parser.add_argument("--foldx-executable", dest="foldx_executable",
                        help="Location of foldx executable")

    options = parser.parse_args()

    pdb_filenames = sorted(glob.glob(os.path.join(options.pdb_input_dir, "*.pdb")))

    for pdb_filename in pdb_filenames:

        foldx_repair_pdb(pdb_filename, options.pdb_output_dir, options.foldx_executable)

        sys.exit()
