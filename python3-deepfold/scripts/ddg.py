from train import get_data_dir, get_name, get_best_checkpoint

def generate_ddg_cmd(ddg_name,
                     model,
                     model_output_type,
                     data_name,
                     regularization,
                     learning_rate,
                     subbatch_size,
                     num_passes,
                     id_number,
                     checkpoint_base_dir,
                     ddg_data_dir_base,
                     output_dir,
                     restore_checkpoint):

    data_dir = get_data_dir(model, ddg_data_dir_base, "")

    name = get_name(model, model_output_type, data_name, regularization, learning_rate, subbatch_size, num_passes, id_number)

    cmd = ['../deepfold_predict_ddg.py',
           '--high-res-input-dir', data_dir,
           '--pdb-dir', ddg_data_dir_base+'/pdbs_reduced',
           '--model-checkpoint-path', checkpoint_base_dir + '/' + name,
           '--ddg-csv-filename', ddg_data_dir_base + '/' + ddg_name + '.csv',
           '--markov-model-filename', ddg_data_dir_base + '/' + 'markov_model',
           '--model', model]

    if restore_checkpoint == 'best':
        best = get_best_checkpoint(checkpoint_base_dir + "/" + name + ".txt")
        cmd += ["--step", best]

    output_file = output_dir + '/' + ddg_name + '_' + restore_checkpoint + "_" + name + '.txt'

    return cmd, output_file, name


if __name__ == '__main__':
    from train import standard_arg_paser, run_cmds, defaults, all_models, all_output_types

    parser = standard_arg_paser(exclude=['mode'])
    args = parser.parse_args()

    # Settings
    checkpoint_base_dir = "/scratch1/rwt891/data/deepfold/camara"
    output_dir = "/scratch1/rwt891/data/deepfold/camara_ddg"

    ddg_data_dir_base = "/scratch1/rwt891/data/ddgs"

    train_data_name = "pc30"
    train_num_passes = "10"
    train_id_number = "01"

    output_type = 'aa'

    # Generate commands
    cmds = []

    for ddg_name in ['curatedprotherm', 'guerois', 'kellogg', 'potapov']:
        for model in all_models:
            cmd = generate_ddg_cmd(ddg_name, model, output_type, train_data_name, defaults['regularization'],
                                   defaults['learning_rate'], defaults['subbatch_size'], train_num_passes,
                                   train_id_number, checkpoint_base_dir, ddg_data_dir_base, output_dir,
                                   restore_checkpoint=args.restore_checkpoint)
            cmds.append(cmd)

    run_cmds(cmds=cmds, devices=['0', '1', '2', '3'], dry=args.dry)
