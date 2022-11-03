import os
import pandas as pd
import numpy as np

class_mapping = {
    'bird': 0,
    'car': 1,
    'dog': 2,
    'lizard': 3,
    'turtle': 4
}

def image_process(path, num_tr=150, num_ts=30):
    img_class = os.path.split(path)[1]
    imgs = os.listdir(path)
    img_set = [os.path.join(path, img) for img in imgs]
    train_set = img_set[: num_tr]
    test_set = img_set[num_tr : num_tr + num_ts]
    with open('imgset/train.txt', 'a+') as f:
        np.savetxt(f, train_set, fmt='%s')
    with open('imgset/test.txt', 'a+') as f:
        np.savetxt(f, test_set, fmt='%s')
    gt_path = os.path.join(os.path.split(path)[0], img_class+'_gt.txt')
    label_process(gt_path, img_class)

def label_process(path, img_cls, num_tr=150, num_ts=30):
    lb = pd.read_csv(path, header=None, index_col=0, sep=' ').to_numpy()
    c = np.ones((lb.shape[0],)) * class_mapping[img_cls]

    x_center = (lb[:, 0] + lb[:, 2]) / 2
    y_center = (lb[:, 1] + lb[:, 3]) / 2
    w = lb[:, 2] - lb[:, 0]
    h = lb[:, 3] - lb[:, 1]
    lb = np.column_stack([x_center, y_center, w, h]) / 128
    lb = np.column_stack([c, lb])
    train_label = lb[: num_tr]
    test_label = lb[num_tr : num_tr + num_ts]

    with open('label/train.txt', 'a+') as f:
        np.savetxt(f, train_label)
    with open('label/test.txt', 'a+') as f:
        np.savetxt(f, test_label)



def process_data():
    BASE = 'tiny_vid'
    dirs = os.listdir(BASE)
    for dir in dirs:
        PATH = os.path.join(BASE,dir)
        if os.path.isdir(PATH):
            image_process(PATH)
            


if __name__ == '__main__':
    process_data()