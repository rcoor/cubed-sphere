def torusdbn_Q_accuracy(batch):
    y = batch["model_output"]
    y_argmax = np.argmax(y, 1)
    predictions = batch["torusdbn"]
    identical = (predictions == y_argmax)
    return np.mean(identical), identical

def pssm_Q_accuracy(batch):
    y = batch["model_output"]
    y_argmax = np.argmax(y, 1)
    predictions = batch["pssm"]
    identical = (np.argmax(predictions, 1) == y_argmax)
    return np.mean(identical), identical

torus_skip_list = ["5COY_", "4Y9I_", "4YSL_", "5HZ7_", "5KVC_", "5A9P_"]

if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    from Deepfold.Models import models
    from Deepfold.batch_factory import BatchFactory
    from utils import str2bool

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--high-res-input-dir", dest="high_res_features_input_dir",
                        help="Location of input files containing high-res features")
    parser.add_argument("--test-set-fraction",
                        help="Fraction of data set aside for testing (default: %(default)s)", type=float, default=0.25)
    parser.add_argument("--validation-set-size",
                        help="Size of validation set (taken out of training set) (default: %(default)s)", type=int, default=10)
    parser.add_argument("--optimize-using-lbfgs", action="store_true",
                        help="Whether to use the LBFGS optimizer")
    parser.add_argument("--num-passes",
                        help="Number of passes over the data during traning (default: %(default)s)", type=int, default=10)
    parser.add_argument("--max-batch-size",
                        help="Maximum batch size used during training (default: %(default)s)", type=int, default=1000)
    parser.add_argument("--subbatch-max-size",
                        help="Maximum batch size used for gradient calculation (default: %(default)s)", type=int, default=25)
    parser.add_argument("--model-checkpoint-path",
                        help="Where to dump/read model checkpoints (default: %(default)s)", default="models")
    parser.add_argument("--max-to-keep",
                        help="Maximal number of checkpoints to keep (default: %(default)s)", type=int, default=2)
    parser.add_argument("--read-from-checkpoint", action="store_true",
                        help="Whether to read model from checkpoint")
    parser.add_argument("--mode", choices=['train', 'test'], default="train", 
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--model-output-type", choices=['aa', 'ss'], default="ss", 
                        help="Whether the model should output secondary structure or amino acid labels (default: %(default)s)")
    parser.add_argument("--dropout-keep-prob", type=float, default=0.5, 
                        help="Probability for leaving out node in dropout (default: %(default)s)")
    parser.add_argument("--learning-rate",
                        help="Learing rate for Adam (default: %(default)s)", type=float, default=0.001)
    parser.add_argument("--reg-fact",
                        help="Regularisation factor (default: %(default)s)", type=float, default=0.001)
    parser.add_argument("--output-interval",
                        help="The output interval for train and validation error  (default: %(default)s)", type=int, default=None)
    parser.add_argument("--model", choices=list(models.keys()), default="CubedSphereModel01",
                        help="Which model definition to use (default: %(default)s)")
    parser.add_argument("--torusdbn-prediction-input-dir", dest="torusdbn_prediction_input_dir",
                        help="Add torusdbn aa and ss predictions to the batch factory. In test mode, comparisons to torusdbn will be printed")
    parser.add_argument("--pssm-input-dir", dest="pssm_input_dir",
                        help="Add evolutionary AA pssm information. In test mode, comparisons to this pssm will be printed")
    parser.add_argument("--step", type=int, default=None,
                        help="Which checkpoint file to use (default: %(default)s)")
    parser.add_argument("--duplicate-origin",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to duplicate the atoms in all bins at the origin for the spherical model")
    
    options = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(options).items()):
        print(key, "=", value)

    high_res_protein_feature_filenames = sorted(glob.glob(os.path.join(options.high_res_features_input_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(glob.glob(os.path.join(options.high_res_features_input_dir, "*residue_features.npz")))

    train_start = 0
    validation_end = test_start = int(len(high_res_protein_feature_filenames)*(1.-options.test_set_fraction))
    train_end = validation_start = int(validation_end-options.validation_set_size)
    test_end = len(high_res_protein_feature_filenames)

    print("# Data:")
    print("Total size: ", len(high_res_protein_feature_filenames))
    print("Training size: ", train_end - train_start)
    print("Validation size: ", validation_end - validation_start)
    print("Test size: ", test_end - test_start)
    
    if options.mode == 'train':
        batch_factory = BatchFactory()
        batch_factory.add_data_set("high_res",
                                   high_res_protein_feature_filenames[:train_end],
                                   high_res_grid_feature_filenames[:train_end],
                                   duplicate_origin=options.duplicate_origin)
        batch_factory.add_data_set("model_output",
                                   high_res_protein_feature_filenames[:train_end],
                                   key_filter=[options.model_output_type+"_one_hot"])

        validation_batch_factory = BatchFactory()
        validation_batch_factory.add_data_set("high_res",
                                              high_res_protein_feature_filenames[validation_start:validation_end],
                                              high_res_grid_feature_filenames[validation_start:validation_end],
                                              duplicate_origin=options.duplicate_origin)
        validation_batch_factory.add_data_set("model_output",
                                              high_res_protein_feature_filenames[validation_start:validation_end],
                                              key_filter=[options.model_output_type+"_one_hot"])
    elif options.mode == 'test':
        batch_factory = BatchFactory()
        batch_factory.add_data_set("high_res",
                                   high_res_protein_feature_filenames[test_start:],
                                   high_res_grid_feature_filenames[test_start:],
                                   duplicate_origin=options.duplicate_origin)
        batch_factory.add_data_set("model_output",
                                   high_res_protein_feature_filenames[test_start:],
                                   key_filter=[options.model_output_type+"_one_hot"])

    if options.torusdbn_prediction_input_dir:
        torusdbn_feature_filenames = []
        for filename in high_res_protein_feature_filenames:
            basename = os.path.basename(filename)
            if not basename[:5] in torus_skip_list:
                torusdbn_feature_filenames.append(os.path.join(options.torusdbn_prediction_input_dir, basename))
        batch_factory.add_data_set("torusdbn",
                                   torusdbn_feature_filenames[test_start:test_end],
                                   key_filter=[options.model_output_type])
    if options.pssm_input_dir:
        pssm_feature_filenames = []
        for filename in high_res_protein_feature_filenames:
            basename = os.path.basename(filename)
            pssm_feature_filenames.append(os.path.join(options.pssm_input_dir, basename))
        batch_factory.add_data_set("pssm",
                                   pssm_feature_filenames[test_start:test_end],
                                   key_filter=["pssm"])

    high_res_grid_size = batch_factory.next(1, increment_counter=False)[0]["high_res"].shape
    output_size        = batch_factory.next(1, increment_counter=False)[0]["model_output"].shape[1]

    if options.model.startswith("Spherical"):
        model = models[options.model](r_size_high_res         = high_res_grid_size[1],
                                      theta_size_high_res     = high_res_grid_size[2],
                                      phi_size_high_res       = high_res_grid_size[3],
                                      channels_high_res       = high_res_grid_size[4],
                                      output_size             = output_size,
                                      optimize_using_lbfgs    = options.optimize_using_lbfgs,
                                      reg_fact                = options.reg_fact,
                                      learning_rate           = options.learning_rate,
                                      model_checkpoint_path   = options.model_checkpoint_path,
                                      max_to_keep             = options.max_to_keep)
    elif options.model.startswith("CubedSphere"):
        model = models[options.model](patches_size_high_res   = high_res_grid_size[1],
                                      r_size_high_res         = high_res_grid_size[2],
                                      xi_size_high_res        = high_res_grid_size[3],
                                      eta_size_high_res       = high_res_grid_size[4],
                                      channels_high_res       = high_res_grid_size[5],
                                      output_size             = output_size,
                                      optimize_using_lbfgs    = options.optimize_using_lbfgs,
                                      reg_fact                = options.reg_fact,
                                      learning_rate           = options.learning_rate,
                                      model_checkpoint_path   = options.model_checkpoint_path,
                                      max_to_keep             = options.max_to_keep)
    elif options.model.startswith("Cartesian"):
        model = models[options.model](x_size_high_res         = high_res_grid_size[1],
                                      y_size_high_res         = high_res_grid_size[2],
                                      z_size_high_res         = high_res_grid_size[3],
                                      channels_high_res       = high_res_grid_size[4],
                                      output_size             = output_size,
                                      optimize_using_lbfgs    = options.optimize_using_lbfgs,
                                      reg_fact                = options.reg_fact,
                                      learning_rate           = options.learning_rate,
                                      model_checkpoint_path   = options.model_checkpoint_path,
                                      max_to_keep             = options.max_to_keep)
    else:
        raise argparse.ArgumentTypeError("Model type not suppported: %s" % options.model)
            

    if options.read_from_checkpoint:
        model.restore(options.model_checkpoint_path, step=options.step)
    
    
    if options.mode == 'train':
        model.train(train_batch_factory      = batch_factory,
                    validation_batch_factory = validation_batch_factory,
                    num_passes               = options.num_passes,
                    max_batch_size           = options.max_batch_size,
                    subbatch_max_size        = options.subbatch_max_size,
                    dropout_keep_prob        = options.dropout_keep_prob,
                    output_interval          = options.output_interval)

    elif options.mode == 'test':

        additional_labels = []
        if options.torusdbn_prediction_input_dir:
            additional_labels.append("TORUSDBN")
        if options.pssm_input_dir:
            additional_labels.append("PSSM")
        print("# Q%s loss PDB %s" % (output_size, " ".join(additional_labels)))

        prev_pdb_id = None
        pdb_ids = set()
        all_identical = np.array([])
        all_entropies = np.array([])

        torusdbn_all_identical = np.array([])
        pssm_all_identical = np.array([])

        more_data = True

        while more_data:

            batch, subbatch_sizes = batch_factory.next(options.max_batch_size,
                                                       subbatch_max_size=options.subbatch_max_size,
                                                       enforce_protein_boundaries=True,
                                                       include_pdb_ids=True,
                                                       return_single_proteins=True)
            more_data = (batch_factory.feature_index != 0)
            loss, identical, entropies, regularization = model.Q_accuracy_and_loss(batch, subbatch_sizes, return_raw=True)

            # Note that the return_single_proteins make sure that the batch always returns a whole protein
            assert(batch["pdb"][1:] == batch["pdb"][:-1])
            assert(batch["pdb"][0] != prev_pdb_id)
            prev_pdb_id = batch["pdb"][0]

            # Update the overall stats
            pdb_ids = pdb_ids.union(set(batch["pdb"]))
            all_identical = np.concatenate((all_identical, identical))
            all_entropies = np.concatenate((all_entropies, entropies))

            # Calculate the accuracy for TorusDBN
            if options.torusdbn_prediction_input_dir:
                if batch["pdb"][0] not in torus_skip_list:
                    torusdbn_Q_test, torusdbn_identical = torusdbn_Q_accuracy(batch)
                    torusdbn_all_identical = np.concatenate((torusdbn_all_identical, torusdbn_identical))
                else:
                    torusdbn_Q_test = None


            # Calculate the accuracy for the PSSM model
            if options.pssm_input_dir:
                pssm_Q_test, pssm_idencical = pssm_Q_accuracy(batch)
                pssm_all_identical = np.concatenate((pssm_all_identical, pssm_idencical))

            # Print the accuracies for this PDB
            additional_values = []
            if options.torusdbn_prediction_input_dir:
                additional_values.append(torusdbn_Q_test)
            if options.pssm_input_dir:
                additional_values.append(pssm_Q_test)

            Q_test = np.mean(identical)
            loss_test = loss

            print(Q_test, loss_test, ",".join([pdb_id for pdb_id in set(batch["pdb"])]), "%s" % " ".join(map(str, additional_values)))

        # Print the overall scores
        Q_test = np.mean(all_identical)
        loss_test = np.mean(all_entropies) + regularization

        print("# Statistics for the whole dataset:")
        print("# Q%s score (test set): %f" % (output_size, Q_test))
        print("# loss (test set): %f" % (loss_test))

        if options.torusdbn_prediction_input_dir:
            print("# TORUSDBN COMPARISON: Q%s score (test set): %s" % (output_size, np.mean(torusdbn_all_identical)))
        if options.pssm_input_dir:
            print("# PSSM COMPARISON: Q%s score (test set): %s" % (output_size, np.mean(pssm_all_identical)))

        print("# PDB IDs: ", [pdb_id for pdb_id in pdb_ids])
