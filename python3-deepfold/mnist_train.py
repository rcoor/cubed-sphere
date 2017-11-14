if __name__ == "__main__":
    import numpy as np
    from MnistCubedSphereModel import MnistCubedSphereModel
    from MnistSphericalModel import MnistSphericalModel

    models = {'CubedSphere': MnistCubedSphereModel,
              'Spherical': MnistSphericalModel}

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(models.keys()), required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('data')
    args = parser.parse_args()

    data = np.load(args.data)

    model = models[args.model]

    if args.model == "CubedSphere":
        cubed_sphere_shape = data['cubed_sphere_grid_matrix_train'].shape
        y_shape = data['y_train'].shape

        mnist_cubed_sphere_model = MnistCubedSphereModel(
            0.01,
            patch_shape=cubed_sphere_shape[1],
            r_shape=cubed_sphere_shape[2],
            xi_shape=cubed_sphere_shape[3],
            eta_shape=cubed_sphere_shape[4],
            output_size=y_shape[1])

        mnist_cubed_sphere_model.train(data['cubed_sphere_grid_matrix_train'], data['y_train'], epochs=args.epochs)
        print("Test error:", mnist_cubed_sphere_model.calc_accuracy(data['cubed_sphere_grid_matrix_test'], data['y_test']))

    elif args.model == "Spherical":
        spherical_shape = data['spherical_grid_matrix_train'].shape
        y_shape = data['y_train'].shape

        mnist_spherical_model = MnistSphericalModel(
            0.01,
            r_shape=spherical_shape[1],
            theta_shape=spherical_shape[2],
            phi_shape=spherical_shape[3],
            output_size=y_shape[1])

        mnist_spherical_model.train(data['spherical_grid_matrix_train'], data['y_train'], epochs=args.epochs)
        print("Test error:", mnist_spherical_model.calc_accuracy(data['spherical_grid_matrix_test'],
                                                                    data['y_test']))