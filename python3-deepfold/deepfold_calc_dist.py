


if __name__ == '__main__':
    import deepfold_model

    from deepfold_models import models
    from utils import str2bool
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--high-res-input-dir", dest="high_res_features_input_dir",
                        help="Location of input files containing high-res features")
    parser.add_argument("--low-res-input-dir", dest="low_res_features_input_dir",
                        help="Location of input files containing low-res features")
    parser.add_argument("--include-high-res-model", dest="include_high_res_model",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to include atomic-resolution part of model")
    parser.add_argument("--include-low-res-model", dest="include_low_res_model",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to include low resolution part of model")
    parser.add_argument("--pdb-dir", dest="pdb_dir",
                        help="Location of PDB files (only used to check residue-index offset)")
    parser.add_argument("--max-batch-size", dest="max_batch_size",
                        help="Maximum batch size used during model evaluation", type=int, default=25)
    parser.add_argument("--model-checkpoint-path", dest="model_checkpoint_path",
                        help="Where to dump/read model checkpoints", default="models")
    parser.add_argument("--ddg-csv-filename", dest="ddg_csv_filename",
                        help="CSV file containing ddG data")
    parser.add_argument("--markov-model-filename", dest="markov_model_filename",
                        help="Parameter file for Markov model")
    parser.add_argument("--model", choices=['default']+list(models.keys()), default="default",
                        help="Which model definition to use (default: %(default)s)")
    parser.add_argument("--step", type=int, default=None,
                        help="Which checkpoint file to use (default: %(default)s)")
    options = parser.parse_args()

    if options.model == "default":
        model = deepfold_model.Model(r_size_high_res         = 24,
                                     theta_size_high_res     = 76,
                                     phi_size_high_res       = 151,
                                     channels_high_res       = 2,
                                     r_size_low_res          = 6,
                                     theta_size_low_res      = 19,
                                     phi_size_low_res        = 38,
                                     channels_low_res        = 43,
                                     output_size             = 20,
                                     max_batch_size          = options.max_batch_size,
                                     max_gradient_batch_size = options.max_batch_size,
                                     include_high_res_model  = options.include_high_res_model,
                                     include_low_res_model   = options.include_low_res_model,
                                     model_checkpoint_path   = options.model_checkpoint_path)
    if options.model.startswith("Spherical"):
        model = models[options.model](r_size_high_res         = 25,
                                      theta_size_high_res     = 76,
                                      phi_size_high_res       = 151,
                                      channels_high_res       = 2,
                                      output_size             = 20,
                                      optimize_using_lbfgs    = False,
                                      reg_fact                = 0.001,
                                      learning_rate           = 0.0001,
                                      model_checkpoint_path   = "/dev/null",
                                      max_to_keep             = 0)
    elif options.model.startswith("CubedSphere"):
        model = models[options.model](patches_size_high_res   = 6,
                                      r_size_high_res         = 24,
                                      xi_size_high_res        = 38,
                                      eta_size_high_res       = 38,
                                      channels_high_res       = 2,
                                      output_size             = 20,
                                      optimize_using_lbfgs    = False,
                                      reg_fact                = 0.001,
                                      learning_rate           = 0.0001,
                                      model_checkpoint_path   = "/dev/null",
                                      max_to_keep             = 0)
    else:
        raise argparse.ArgumentTypeError("Deep model not supported: %s" % parser.deep_mode)
        

    model.restore(options.model_checkpoint_path, step=options.step)
