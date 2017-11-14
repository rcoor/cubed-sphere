if __name__ == "__main__":

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Deepfold import grid

    ### 3D plot ###
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    c = lambda t, a: np.array([np.cos(t), np.sin(t), -a*t]) / np.sqrt(1 + a**2 * t**2)
    p = np.transpose(np.array([c(t, .1) for t in np.linspace(-40, 40, 1000)]))
    #p = np.transpose(np.array([c(t, .01) for t in np.linspace(-400, 40, 1000)]))

    #p = p[[0, 2, 1], :]
    #p = p[[2, 0, 1], :]
        
    ax.plot(p[0, :], p[1, :], p[2, :])

    fig.savefig("plot_3d.png")
    plt.close()

    ### 2D cubed sphere plot ###
    
    p_cubed = [grid.cartesian_to_cubed_sphere(*v) for v in np.transpose(p)]
    p_unfolded_plane = np.array([grid.cubed_sphere_to_unfolded_plane(v[0], v[2], v[3]) for v in p_cubed])

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for v in grid._offsets:
        ax.add_patch(matplotlib.patches.Rectangle(v, 1., 1., fill=False))

    colors = plt.get_cmap("hsv")(np.linspace(0, 1, p_unfolded_plane.shape[0]))
        
    for i in range(p_unfolded_plane.shape[0]-1):
        if p_cubed[i][0] == p_cubed[i+1][0]:
            linestyle = '-'
        else:
            linestyle = ':'

        ax.plot(p_unfolded_plane[i:i+2, 0], p_unfolded_plane[i:i+2, 1], color=colors[i], linestyle=linestyle)

    plt.axis('off')
    e = 0.1
    ax.set_xlim(0-e, 4+e)
    ax.set_ylim(0-e, 3+e)

    fig.savefig("plot_2d.png")
    plt.close()


    ### 2D cubed sphere grid plot ###
    def cubed_sphere_bins_to_unfolded_plane(patch, xi, eta, xi_bins, eta_bins, offsets=grid._offsets):
        return offsets[patch] + np.array([xi, eta], dtype=np.float) / np.array([xi_bins, eta_bins], dtype=np.float)

    r_bins = 2
    xi_bins = 100
    eta_bins = 100

    p_bins = [grid.discretize_into_cubed_sphere_bins(patch=v[0], r=v[1], xi=v[2], eta=v[3],  max_r=100,
                                                     r_shape=r_bins, xi_shape=xi_bins, eta_shape=eta_bins)
              for v in p_cubed]

    p_bin_unfolded_plane = np.array([cubed_sphere_bins_to_unfolded_plane(v[0], v[2], v[3], xi_bins, eta_bins) for v in p_bins])

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for v in grid._offsets:
        ax.add_patch(matplotlib.patches.Rectangle(v, 1., 1., fill=False))

    for i in range(p_bin_unfolded_plane.shape[0]-1):
        if p_cubed[i][0] == p_cubed[i+1][0]:
            linestyle = '-'
        else:
            linestyle = ':'

        ax.plot(p_bin_unfolded_plane[i:i+2, 0], p_bin_unfolded_plane[i:i+2, 1], color=colors[i], linestyle=linestyle)

    plt.axis('off')
    e = 0.1
    ax.set_xlim(0-e, 4+e)
    ax.set_ylim(0-e, 3+e)

    fig.savefig("plot_2d_bins.png")
    plt.close()


    ### Convert to matrix and replot ###
    def my_imshow(ax, X, xi_padding=None, eta_padding=None):
        if xi_padding is not None and eta_padding is not None:
            ax.add_patch(matplotlib.patches.Rectangle([xi_padding, eta_padding], X.shape[0]-2*xi_padding, X.shape[1]-2*eta_padding, fill=False))

        idx = np.where(X>0)
        c = [colors[int(i)-1] for i in X[idx]]
        ax.scatter(idx[0], idx[1], s=1.0, marker=".", color=c)

    grid_matrix = np.zeros((6, r_bins, xi_bins, eta_bins, 1))

    for i, (patch, r_bin, xi_bin, eta_bin) in enumerate(p_bins):
        grid_matrix[patch, r_bin, xi_bin, eta_bin, 0] = 1 + i

    fig = plt.figure()

    for patch in range(grid_matrix.shape[0]):
        idx = {4:2, 3:5, 0:6, 1:7, 2:8, 5:10}[patch]

        ax = fig.add_subplot(3, 4, idx, aspect='equal')
        my_imshow(ax, grid_matrix[patch, 0, :, :, 0])
        plt.axis('off')
        
    fig.savefig("plot_2d_grid.png")
    plt.close()

    ### Add padding and replot ###
    import tensorflow as tf
    from deepfold_model_cubed_sphere import Model


    
    X = tf.placeholder(tf.float32, [None, 6, r_bins, xi_bins, eta_bins, 1])
    pad = 50
    
    X_pad = Model.pad_cubed_sphere_grid(X, r_padding=(0,0), xi_padding=(pad, pad), eta_padding=(pad, pad))

    with tf.Session().as_default() as session:
        grid_matrix_padded = session.run(X_pad, feed_dict={X:grid_matrix[None]})

    fig = plt.figure()

    for patch in range(grid_matrix_padded.shape[1]):
        idx = {4:2, 3:5, 0:6, 1:7, 2:8, 5:10}[patch]

        ax = fig.add_subplot(3, 4, idx, aspect='equal')
        my_imshow(ax, grid_matrix_padded[0, patch, 0, :, :, 0], xi_padding=pad, eta_padding=pad)
        
        #ax.imshow(grid_matrix_padded[0, patch, 0, :, :, 0].T, aspect='equal', origin='lower', cmap='gray')
        plt.axis('off')
        
    fig.savefig("plot_2d_grid_padded.pdf")
    plt.close()
        
                
    

    

    

