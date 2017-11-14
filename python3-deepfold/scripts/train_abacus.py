script_pre="""#!/bin/bash
#
#SBATCH --account itucs_gpu       # account
#SBATCH --nodes 1                 # number of nodes
#SBATCH --time 24:00:00           # max time (HH:MM:SS)

"""


script_post = """
wait
"""


if __name__ == '__main__':
    from train import standard_arg_paser, generate_train_cmd, defaults, all_output_types
    
    parser = standard_arg_paser()
    args = parser.parse_args()

    # Settings
    checkpoint_base_dir = "/scratch/itucs/frellsen/data/deepfold/camara"
    test_base_dir = "/scratch/itucs/frellsen/data/deepfold/camara_test"

    id_number = "a01"

    # Print commands
    for data_dir_base, data_name, num_passes in [("/work/itucs/frellsen/data/cull_pdb_pc100_entries_170602/culled_pc30_res3.0_R0.3_d170611/",
                                                  "pc30",
                                                  "10"),
                                                 ("/work/itucs/frellsen/data/cull_pdb_pc100_entries_170602/",
                                                  "pc100",
                                                  "5")
                                                 ]:


    
        for output_type in all_output_types:
            model = 'CubedSphereHighRModel'

            cmd = generate_train_cmd(args.mode, model, output_type, data_name, defaults['regularization'],
                                     defaults['learning_rate'], defaults['subbatch_size'], num_passes, id_number,
                                     data_dir_base, checkpoint_base_dir, test_base_dir,
                                     prefix="CUDA_VISIBLE_DEVICES=0 python -u /home/frellsen/Projects/deepfold-master/")

            script_contents = script_pre + ' '.join(cmd[0]) + " &> " + cmd[1] + "\n" + script_post
            
            with open("/home/frellsen/Scripts/" + cmd[2] + ".bash", 'w') as script_file:
                script_file.write(script_contents)
