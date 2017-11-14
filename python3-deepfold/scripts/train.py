import os
import subprocess
import time
import argparse


def run_cmds(cmds, devices, dry, force):
    # Check if output files exist
    for cmd, output_file, name in cmds:
        if os.path.isfile(output_file):
            print("Output file already exists: " + output_file)

            if not force:
                exit()

    # Run the commands
    device_used_by = dict(list(zip(devices, [None] * len(devices))))

    for cmd, output_file, name in cmds:
        # Check if there is a free device
        free_device = None

        while free_device is None:
            for device in device_used_by:
                if device_used_by[device] is None or device_used_by[device].poll() is not None:
                    free_device = device
                    break

            if free_device is not None:
                break
            else:
                time.sleep(5)

        full_cmd = ["python", "-u"] + cmd

        print("Device:", device)
        print("Starting:", ' '.join(full_cmd))
        print("Output:", output_file)
        print()

        if not dry:
            output_fh = open(output_file, "w")
            process = subprocess.Popen(full_cmd,
                                       stdout=output_fh,
                                       stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=device))
        else:
            process = None

        device_used_by[device] = process

    for process in device_used_by.values():
        if process is not None:
            process.wait()


def get_data_dir(model, data_dir_base, data_dir_postfix):
    if model.startswith("CubedSphere"):
        data_dir = data_dir_base + "/atomistic_features_cubed_sphere" + data_dir_postfix
    elif model.startswith("CartesianHighres"):
        data_dir = data_dir_base + "/atomistic_features_cartesian_highres" + data_dir_postfix
    elif model.startswith("Cartesian"):
        data_dir = data_dir_base + "/atomistic_features_cartesian" + data_dir_postfix
    elif model.startswith("Spherical"):
        data_dir = data_dir_base + "/atomistic_features_spherical" + data_dir_postfix
    else:
        raise ValueError("Unknown model: " + model)

    return data_dir


def get_best_checkpoint(log_file):
    return subprocess.check_output(['python', '../qmax.py', '-l', log_file]).strip()


def get_name(model, model_output_type, data_name, regularization, learning_rate, subbatch_size, num_passes, id_number):
    return "_".join([model, model_output_type, data_name, "reg" + regularization.replace(".", ""),
                     "lr" + learning_rate.replace(".", ""), "sb" + subbatch_size, "np" + num_passes, "id" + id_number])


def generate_train_cmd(mode,
                       model,
                       model_output_type,
                       data_name,
                       regularization,
                       learning_rate,
                       subbatch_size,
                       num_passes,
                       id_number,
                       data_dir_base,
                       checkpoint_base_dir,
                       test_base_dir,
                       restore_checkpoint,
                       prefix="../",
                       data_dir_postfix=""):

    data_dir = get_data_dir(model, data_dir_base, data_dir_postfix)

    name = get_name(model, model_output_type, data_name, regularization, learning_rate, subbatch_size, num_passes,
                    id_number)

    cmd = [prefix + "deepfold_train.py",
           "--high-res-input-dir", data_dir,
           "--subbatch-max-size", subbatch_size,
           "--reg-fact", regularization,
           "--learning-rate", learning_rate,
           "--model-output-type", model_output_type,
           "--model", model,
           "--duplicate-origin", "false",
           "--mode", mode,
           "--num-passes", num_passes,
           "--output-interval", "100",
           "--max-to-keep", "0",
           "--model-checkpoint-path", checkpoint_base_dir + "/" + name]

    if mode == "train":
        output_file = checkpoint_base_dir + "/" + name + ".txt"
        cmd += ["--max-batch-size", "1000"]

    elif mode == "test":
        output_file = test_base_dir + "/test_" + restore_checkpoint + "_" + name + ".txt"
        cmd += ["--read-from-checkpoint",
                "--max-batch-size", "100000"]

        if restore_checkpoint == 'best':
            best = get_best_checkpoint(checkpoint_base_dir + "/" + name + ".txt")
            cmd += ["--step", best]

        if model.startswith("Cartesian"):
            if os.path.isdir(data_dir_base + '/torusdbn_predictions'):
                cmd += ["--torusdbn-prediction-input-dir", data_dir_base + '/torusdbn_predictions']

            if os.path.isdir(data_dir_base + '/residue_features_cartesian') and model_output_type=='aa':
                cmd += ["--pssm-input-dir", data_dir_base + '/residue_features_cartesian']

    return cmd, output_file, name


def standard_arg_paser(include=['f', 'mode', 'v', 'dry', 'restore-checkpoint'], exclude=[]):
    include = set(include) - set(exclude)
    parser = argparse.ArgumentParser()
    if 'f' in include:
        parser.add_argument("-f", dest="force", help="force", action="store_true")
    if 'mode' in include:
        parser.add_argument("--mode", choices=['train', 'test'], required=True)
    if 'v' in include:
        parser.add_argument("-v", dest="verbose", help="vervose", action="store_true")
    if 'dry' in include:
        parser.add_argument("--dry", help="dry run", action="store_true")
    if 'restore-checkpoint' in include:
        parser.add_argument("--restore-checkpoint", choices=['last', 'best'], default='best')
    return parser


defaults = dict(
    regularization="0.001",
    learning_rate="0.0001",
    subbatch_size="100"
)

all_models = ["CubedSphereModel", "CubedSphereBandedModel", "CubedSphereBandedDisjointModel", "CubedSphereDenseModel",
              "SphericalModel", "CartesianHighresModel"]
all_output_types = ["aa", "ss"]

if __name__ == '__main__':
    parser = standard_arg_paser()
    args = parser.parse_args()

    # Settings
    checkpoint_base_dir = "/scratch1/rwt891/data/deepfold/camara"

    data_dir_base = "/scratch1/rwt891/data/cull_pdb_pc100_entries_170602/culled_pc30_res3.0_R0.3_d170611"
    data_name = "pc30"
    num_passes = "10"

    # data_dir_base = "/scratch1/rwt891/data/cull_pdb_pc100_entries_170602/"
    # data_name = "pc100"
    # num_passes = "5"

    id_number = "01"

    test_base_dir = "/scratch1/rwt891/data/deepfold/camara_test2"

    # Generate commands
    cmds = []

    for output_type in all_output_types:
        #for model in all_models:
        for model in ["CartesianHighresModel"]:
            cmd = generate_train_cmd(args.mode, model, output_type, data_name, defaults['regularization'],
                                     defaults['learning_rate'], defaults['subbatch_size'], num_passes, id_number,
                                     data_dir_base, checkpoint_base_dir, test_base_dir,
                                     restore_checkpoint=args.restore_checkpoint)
            cmds.append(cmd)

    run_cmds(cmds=cmds, devices=["0", "1"], dry=args.dry, force=args.force)
