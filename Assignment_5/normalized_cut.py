from cluster_algorithm import ClusterAlgorithm

class NormalizedCut(ClusterAlgorithm):
    """docstring for ClusterAlgorithm"""
    def __init__(self, gamma, k_means):
        self.gamma = gamma
        self.k_means = k_means

    def get_segmentations(self, image):
        old_x, old_y = image.shape[0], image.shape[1]
        image = np.reshape(x_train[0], (image.shape[0]*image.shape[1],image.shape[2]))
        k_segmentations = []
        for k in k_means:
            #TODO
            # self.gamma
            # segmentation =
            k_segmentations.append(segmentation)

        return k_segmentations