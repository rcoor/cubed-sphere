if __name__ == "__main__":
    import numpy as np
    from mnist_data import img_batch_to_XYZ, XYZ_batch_to_spherical_batch, XYZ_batch_to_cubed_sphere_batch


    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)

    X, y = mnist.train.next_batch(1, shuffle=True)
    print(X.shape, y.shape)

    XYZ, color = img_batch_to_XYZ(X, density=1, z_coord=1, random_rotation=False, center='y')

    #spherical_grid_matrix = XYZ_batch_to_spherical_batch(XYZ, color)
    #cubed_shape_grid_matrix = XYZ_batch_to_cubed_sphere_batch(XYZ, color, xi_shape=28, eta_shape=28)


    # Represent the picture as 3D points
    X = X[0,:]

    Xc = []
    Xd = []


    for i in range(28):
        for j in range(28):
            xc = np.array([i/28., j/28., 1]) * 2 - 1
            xc /= np.sqrt((xc**2).sum())

            xd = X[i + 28*j]

            Xc.append(xc)
            Xd.append(xd)





    Xc = np.array(Xc).T






    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.art3d as art3d

    fig = plt.figure()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])

    from matplotlib import cm

    ax.scatter(Xc[0, :], Xc[1, :], Xc[2, :]+1, c=Xd)
    ax.scatter(XYZ[0,:, 0], XYZ[0,:, 1], XYZ[0,:, 2], cmap=cm.Greys, c=color[0,:])

    # Plot a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    plt.show()

    # x = (2*X[:,0])/(1+np.squared(X[:,0]) + np.squared(X[:,1]))
    # y = (2*X[:,1])/(1+np.squared(X[:,0]) + np.squared(X[:,1]))
    # z = (-1+2*X[:,0]+2*X[:,1])/(1+np.squared(X[:,0]) + np.squared(X[:,1]))






    # plt.imshow(X)
    # plt.show()
