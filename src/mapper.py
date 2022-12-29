import numpy as np
import networkx as nx
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

def select_k(data, max_k=20, alpha = .1):
    """
    method from https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c
    """
    scaled_inertia = list()
    if max_k == 1:
        return 1
    else:
        kmeans = MiniBatchKMeans(n_clusters=1)
        kmeans.fit(data)
        inertia = kmeans.inertia_
        scaled_inertia.append(inertia)
        for k in range(2, max_k):
            kmenas = MiniBatchKMeans(n_clusters=k)
            kmeans.fit(data)

            scaled_inertia.append(kmeans.inertia_/inertia + alpha*k)

        scaled_inertia = np.array(scaled_inertia)

        return scaled_inertia.argmin() + 1


class Mapper():
    def __init__(self, data, projected_data, percentile = 10, perc_overlap = 3):
        self.data = data
        self.projected_data = projected_data
        self.percentile = percentile
        self.perc_overlap = perc_overlap

        self.df = pd.DataFrame(index=np.arange(len(data)))

        # to fit
        self.bins = None
        self.graph = None

    def overlapping_bins(self):

        top = np.percentile(self.projected_data, np.arange(0, 100, self.percentile) + self.perc_overlap)
        bottom = np.percentile(self.projected_data, np.arange(10, 100, self.percentile) - self.perc_overlap)

        bins = [[self.projected_data.min(), top[1]]]
        for i in np.arange(0,100//self.percentile-1):
            bins.append([bottom[i], top[i+1]])
        bins.append([bottom[-1], self.projected_data.max()])


        return bins

    def get_preimages(self, bin):
        bottom, top = bin
        indices = np.logical_and(self.projected_data >= bottom,
                                 self.projected_data <= top)
        preimage = self.data[indices]

        return preimage, indices


    def cluster(self, preimage):
        k = select_k(preimage)

        kmeans = MiniBatchKMeans(k)
        kmeans.fit(preimage)

        return kmeans

    def create_graph(self):
        g = nx.Graph()
        bins = self.overlapping_bins()

        for j in range(len(bins)):

            bin = bins[j]
            preimage, indices = self.get_preimages(bin)
            kmeans = self.cluster(preimage)

            for i in set(kmeans.labels_):
                g.add_node((j,i))

            self.df["{}".format(j)] = [np.NaN]*len(self.data)
            self.df.loc[indices, "{}".format(j)] = kmeans.labels_

        return g


    def fit(self):
        self.graph = self.create_graph()
