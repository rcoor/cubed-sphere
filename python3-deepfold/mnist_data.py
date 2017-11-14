import numpy as np
import Deepfold.grid as grid


def rotation_x(theta):
    return np.array([[1,            0,              0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])

def rotation_y(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [             0, 1,             0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]])


def rotation(alpha, beta, gamma):
    return np.dot(np.dot(rotation_z(gamma), rotation_y(beta)), rotation_x(alpha))


def img_batch_to_XYZ(X, density=1, z_coord=2, ms=28, random_rotation=False, center='z'):
    grid_points = np.linspace(-1, 1, density*ms+1)[:-1] + 1./(density*ms)
    grid = np.meshgrid(grid_points, grid_points)
    XYZ = np.vstack(list(map(np.ravel, grid)) + [np.zeros((density*ms)**2)+z_coord]).T
    XYZ /= np.array([np.sqrt((XYZ**2).sum(axis=1))]*3).T

    XYZs = []
    colors = []

    for i in range(X.shape[0]):
        if center=='x':
            pass
            rot = rotation_y(-np.pi/2)
            XYZ = np.dot(XYZ, rot)
        elif center=='y':
            rot = rotation_x(np.pi/2)
            XYZ = np.dot(XYZ, rot)
        elif center=='z':
            pass
        else:
            raise ValueError("Unknown value for center:", center)

        if random_rotation:
            rot = rotation(np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi))
            XYZ = np.dot(XYZ, rot)

        XYZs.append(XYZ)

        assert (density == 1)
        color = X[i]
        # color = np.hstack([np.hstack([X.T.flatten()] * density)] * density).T.flatten()
        # color = np.array([np.array([X.T] * density).flatten()]*density).flatten()
        colors.append(color)

    return np.array(XYZs), np.array(colors)


def XYZ_batch_to_spherical_batch(XYZ, color, theta_shape, phi_shape):
    r_shape = 1
    max_radius = 2

    grid_matrices = []

    for i in range(XYZ.shape[0]):
        # Convert the input points to spherical coordinates and bin them
        r, theta, phi = grid.cartesian_to_spherical_coordinates(XYZ[i, :, :])
        r_bin, theta_bin, phi_bin = grid.discretize_into_spherical_bins(r, theta, phi, max_radius, r_shape, theta_shape, phi_shape)

        indices = np.vstack((r_bin, theta_bin, phi_bin, np.zeros(XYZ.shape[1], dtype=np.int64))).transpose()

        # Create the data matrices
        sum_matrix = np.zeros(shape=(r_shape, theta_shape, phi_shape, 1))
        count_matrix = np.zeros(shape=(r_shape, theta_shape, phi_shape, 1))

        # Calculate the mean intentency in the sum_matrix
        np.add.at(sum_matrix, [indices[:,j] for j in range(indices.shape[1])], color[i,:])
        np.add.at(count_matrix, [indices[:, j] for j in range(indices.shape[1])], 1)
        sum_matrix[count_matrix > 0] /= count_matrix[count_matrix > 0]

        grid_matrices.append(sum_matrix)

    return np.array(grid_matrices)


def XYZ_batch_to_cubed_sphere_batch(XYZ, color, xi_shape=28, eta_shape=28):
    r_shape = 1
    max_radius = 2

    grid_matrices = []

    for i in range(XYZ.shape[0]):
        # Convert the input points to cubed sphere coordinates and bin them
        patch, r, xi, eta = grid.cartesian_to_cubed_sphere_vectorized(XYZ[i, :, 0], XYZ[i, :, 1], XYZ[i, :, 2])
        patch, r_bin, xi_bin, eta_bin = grid.discretize_into_cubed_sphere_bins(patch, r, xi, eta, max_radius, r_shape, xi_shape, eta_shape)

        indices = np.vstack((patch, r_bin, xi_bin, eta_bin, np.zeros(XYZ.shape[1], dtype=np.int64))).transpose()

        # Create the data matrices
        sum_matrix = np.zeros(shape=(6, r_shape, xi_shape, eta_shape, 1))
        count_matrix = np.zeros(shape=(6, r_shape, xi_shape, eta_shape, 1))

        # Calculate the mean intentency in the sum_matrix
        np.add.at(sum_matrix, [indices[:,j] for j in range(indices.shape[1])], color[i,:])
        np.add.at(count_matrix, [indices[:, j] for j in range(indices.shape[1])], 1)
        sum_matrix[count_matrix > 0] /= count_matrix[count_matrix > 0]



        grid_matrices.append(sum_matrix)

    return np.array(grid_matrices)


def chunks(l, n, shuffle=False):
    """
    Yield successive n-sized chunks from l.

    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    assert(type(l)==list)
    assert(len(l)>0)
    assert(len(set(map(len, l)))==1)

    if not shuffle:
        for i in range(0, len(l[0]), n):
            yield [l[j][i:i + n] for j in range(len(l))]
    else:
        perm = np.random.permutation(len(l[0]))
        for i in range(0, len(l[0]), n):
            yield [l[j][perm[i:i + n]] for j in range(len(l))]


def shuffle(l):
    assert(type(l)==list)
    assert(len(l)>0)
    assert(len(set(map(len, l)))==1)

    perm = np.random.permutation(len(l[0]))

    return  [l[j][perm] for j in range(len(l))]


# Wrapped function for parallel functionality
def wrapped_XYZ_batch_to_spherical_batch(l):
    return XYZ_batch_to_spherical_batch(l[0], l[1], theta_shape=2 * 28, phi_shape=4 * 28)

def wrapped_XYZ_batch_to_cubed_sphere_batch(l):
    return XYZ_batch_to_cubed_sphere_batch(l[0], l[1], xi_shape=28, eta_shape=28)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--center', choices=['x', 'y', 'z'], default='z')
    parser.add_argument('--rotate', choices=['t', 'f'], default='f')
    parser.add_argument('--proc', type=int, default=10)
    args = parser.parse_args()

    import multiprocessing
    pool = multiprocessing.Pool(args.proc)

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print('Train:', mnist.train.images.shape, mnist.train.labels.shape)
    print('Test: ', mnist.test.images.shape, mnist.test.labels.shape)

    # Make training data
    X_train, y_train = mnist.train.images, mnist.train.labels

    XYZ_train, color_train = img_batch_to_XYZ(X_train, density=1, z_coord=1, random_rotation=args.rotate=='t', center=args.center)

    # Parallel version of the following command
    #   spherical_grid_matrix_train = XYZ_batch_to_spherical_batch(XYZ_train, color_train, theta_shape=2*28, phi_shape=4*28)
    spherical_grid_matrix_train = pool.map(wrapped_XYZ_batch_to_spherical_batch, chunks([XYZ_train, color_train], 10))
    spherical_grid_matrix_train = np.vstack(spherical_grid_matrix_train)

    # Parallel version of the following command
    #   cubed_sphere_grid_matrix_train = XYZ_batch_to_cubed_sphere_batch(XYZ_train, color_train, xi_shape=28, eta_shape=28)
    cubed_sphere_grid_matrix_train = pool.map(wrapped_XYZ_batch_to_cubed_sphere_batch, chunks([XYZ_train, color_train], 10))
    cubed_sphere_grid_matrix_train = np.vstack(cubed_sphere_grid_matrix_train)

    # Make test data
    X_test, y_test = mnist.test.images, mnist.test.labels

    XYZ_test, color_test = img_batch_to_XYZ(X_test, density=1, z_coord=100, random_rotation=False)

    # Parallel version of the following command
    #  spherical_grid_matrix_test = XYZ_batch_to_spherical_batch(XYZ_test, color_test, theta_shape=2*28, phi_shape=4*28)
    spherical_grid_matrix_test = pool.map(wrapped_XYZ_batch_to_spherical_batch, chunks([XYZ_test, color_test], 10))
    spherical_grid_matrix_test = np.vstack(spherical_grid_matrix_test)

    # Parallel version of the following command
    #  cubed_sphere_grid_matrix_test = XYZ_batch_to_cubed_sphere_batch(XYZ_test, color_test, xi_shape=28, eta_shape=28)
    cubed_sphere_grid_matrix_test = pool.map(wrapped_XYZ_batch_to_cubed_sphere_batch, chunks([XYZ_test, color_test], 10))
    cubed_sphere_grid_matrix_test = np.vstack(cubed_sphere_grid_matrix_test)

    # Save the data
    np.savez_compressed("MNIST_data_sphere/data_center-%s_rotate-%s.npz" % (args.center, args.rotate),
                        spherical_grid_matrix_train=spherical_grid_matrix_train,
                        cubed_sphere_grid_matrix_train=cubed_sphere_grid_matrix_train,
                        y_train=y_train,
                        spherical_grid_matrix_test=spherical_grid_matrix_test,
                        cubed_sphere_grid_matrix_test=cubed_sphere_grid_matrix_test,
                        y_test=y_test)
