import numpy as np
import sklearn.datasets
import random

# set random seeds
seed = 39

np.random.seed(seed)
random.seed(seed)

size = 5000

blobs, y_blobs = sklearn.datasets.make_blobs(n_samples=size, centers = [[0,0],[ 4,4]], cluster_std=1.)
np.save("../data/toy/blobs.npy", np.c_[blobs, y_blobs])

moons, y_moons = sklearn.datasets.make_moons(n_samples=size)
np.save("../data/toy/moons.npy", np.c_[moons, y_moons])

circles, y_circles = sklearn.datasets.make_circles(n_samples=size)
circles *= 10
np.save("../data/toy/circles.npy", np.c_[circles, y_circles])

circles_plus, y_circles_plus =sklearn.datasets.make_circles(n_samples=(size//2 -1000, size//2))
circles_plus *= 10
gauss = np.random.multivariate_normal([-4, 4], .5*np.identity(2), size=1000)

circles_plus = np.concatenate([circles_plus, gauss])
y_circles_plus = np.concatenate([y_circles_plus, np.ones(1000)])
np.save("../data/toy/circles_plus.npy", np.c_[circles_plus, y_circles_plus])