import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
import metrics

class ClusterAlgorithm(object):
    """docstring for ClusterAlgorithm"""
    def __init__(self):
        pass

    def train(self, x_train, y_train, verbose=False):
        for i in range(len(x_train)):
            k_segmentations = self.get_segmentations(x_train[i])
            f_measure, c_measure = self.calc_performance(k_segmentations, y_train[i]['groundTruth'], verbose=verbose)


    def get_segmentations(self, image, k_means):
        pass


    def calc_performance(self, segmentations, ground_truths, verbose=False):
        num_truthes = ground_truths.shape[1]
        f_per_seg = []
        c_per_seg = []
        for i, seg in enumerate(segmentations):
            f = 0
            c=0
            for j in range(num_truthes):
                grnd = ground_truths[:,j][0][0][0][0]
                fm = metrics.f_measure(seg, grnd)
                cond_entropy = v_measure_score(seg.flat, grnd.flat)
                f += fm
                c += cond_entropy
                if verbose:
                    print("k:{} vs truth:{} => f = {}, c = {}".format(i+1,j+1, fm, cond_entropy))
            f /= num_truthes
            c /= num_truthes
            f_per_seg.append(f)
            c_per_seg.append(c)
        f_measure = np.mean(f_per_seg)
        c_measure = np.mean(c_per_seg)
        if verbose:
            print("Avg. f-measue: {} , Avg. conditional entropy: {}".format(f_measure, c_measure))
        return f_measure, c_measure