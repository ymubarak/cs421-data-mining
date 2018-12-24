import numpy as np

def f_measure(seg, grnd):
    clusters = np.unique(seg).tolist()
    F = 0
    for c in clusters:
        indeces = np.where(seg==c)
        # percision
        (values, counts) = np.unique(grnd[indeces], return_counts=True)
        ind = np.argmax(counts)
        max_val = values[ind]
        occurances = counts[ind]
        prec = occurances/ len(indeces[0])
        # recall
        rec = occurances / len(grnd[grnd==max_val])
        f = 2*prec*rec / (prec+rec)
        F += f
    F /= len(clusters)
    return F

def conditional_entropy(seg, grnd):
    clusters = np.unique(seg).tolist()
    values= np.unique(grnd)
    H = 0
    for c in clusters:
        Hi = 0
        indeces = np.where(seg==c)
        c_match = grnd[indeces]
        for i in range(len(values)): # for k in grnd K's
            nij = len(c_match[c_match==values[i]])
            ni =  len(indeces)
            if nij/ni !=0:
                Hi += (nij/ni)*np.log2(nij/ni)
#                 print(Hi)
        Hi *= -1
        H += len(indeces)*Hi/len(seg)
    return H

# conditional_entropy(k_segmentations[0].flat, y_train[0]['groundTruth'][:,0][0][0][0][0].flat)