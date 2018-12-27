from cluster_algorithm import ClusterAlgorithm
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.cluster import SpectralClustering
import cv2

class NormalizedCut(ClusterAlgorithm):
    """docstring for ClusterAlgorithm"""
    def __init__(self, gamma=1, k_means=5, affinity='rbf'):
        self.gamma = gamma
        self.k_means = k_means
        self.affinity = affinity

    def get_segmentations(self, image):
        if self.affinity not in ['rbf','nearest_neighbors']:
            raise Exception("affinity must be rbf or nearest_neighbors only")
        image = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        x, y, z = image.shape[0], image.shape[1], image.shape[2]
        image = np.reshape(image, (x*y, z))
        k_segmentations = []
        for n in self.k_means:
            clustering = SpectralClustering(n_clusters=n, affinity=self.affinity, eigen_tol=0.001, gamma=self.gamma, n_neighbors=5 , n_jobs=-1, eigen_solver='arpack', random_state=0)
            clustering.fit(image)
            segmentation = clustering.labels_.reshape(x, y)
            k_segmentations.append(segmentation)

        return k_segmentations
    
    def calc_performance(self, segmentations, ground_truths, verbose=False):
        num_truthes = len(ground_truths)
        f_per_seg = []
        c_per_seg = []
        for i, seg in enumerate(segmentations):
            k = self.k_means[i]
            f = 0
            c=0
            for j in range(num_truthes):
                gt = cv2.resize(ground_truths[j], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                fm = f_measure(seg, gt)
                cond_entropy = v_measure_score(seg.flat, gt.flat)
                f += fm
                c += cond_entropy
                if verbose:
                    print("k:{} VS gorund-truth:{} => f={:.4f}, c={:.4f}".format(k,j+1, fm, cond_entropy))
            f /= num_truthes
            c /= num_truthes
            f_per_seg.append(f)
            c_per_seg.append(c)
            print("k={}: Avg. f-measue={:.4f} , Avg. conditional entropy={:.4f}\n".format(k, f, c))
        return f_per_seg, c_per_seg