from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot


def gen_anchors(w_h, n_anchors=9, imgsz=128):
    cluster = KMeans(n_clusters=n_anchors)
    cluster.fit(w_h)
    anchors = np.round(cluster.cluster_centers_ * imgsz)
    print(anchors.tolist())
    

def load_gt(path):
    gts = np.loadtxt(path)
    wh = gts[:, -2:]
    return wh

if __name__ == '__main__':
    w_h = load_gt('label/train.txt')
    gen_anchors(w_h, imgsz=256)
    