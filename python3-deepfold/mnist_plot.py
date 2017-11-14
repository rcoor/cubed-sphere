import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X, y = mnist.train.next_batch(1)




# Represent the picture as 3D points

X = X.reshape((28,28))

Xc = []
Xd = []


for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xc = np.array([i/28., j/28., 2]) * 2 - 1
        xc /= np.sqrt((xc**2).sum())

        xd = X[i, j]

        Xc.append(xc)
        Xd.append(xd)







Xc = np.array(Xc).T











print(X.shape)
print(X)











import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])

print(Xc.shape)

ax.scatter(Xc[0, :], Xc[1, :], Xc[2, :], c=Xd)

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
