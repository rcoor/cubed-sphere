if __name__ == '__main__':
    from train import standard_arg_paser, generate_train_cmd, run_cmds, defaults, all_models, all_output_types
    
    parser = standard_arg_paser()
    args = parser.parse_args()

    # Settings
    checkpoint_base_dir = "/scratch1/rwt891/data/deepfold/camara"
    test_base_dir = "/scratch1/rwt891/data/deepfold/camara_test"

    id_number = "01"

    # Generate commands
    cmds = []
    for data_dir_base, data_name, num_passes in [("/scratch1/rwt891/data/cull_pdb_pc100_entries_170602/culled_pc30_res3.0_R0.3_d170611/",
                                                  "pc30",
                                                  "10"),
                                                 ("/scratch1/rwt891/data/cull_pdb_pc100_entries_170602/",
                                                  "pc100",
                                                  "5")
                                                 ]:
        for output_type in all_output_types:
            model = "SphericalModel"
            cmd = generate_train_cmd(args.mode, model, output_type, data_name, defaults['regularization'], defaults['learning_rate'], defaults['subbatch_size'], num_passes, id_number, data_dir_base, checkpoint_base_dir, test_base_dir)
            cmds.append(cmd)

    run_cmds(cmds=cmds, devices=["0", "2", "3"], dry=args.dry)
