import os
import numpy as np 

import argparse

parser = argparse.ArgumentParser('merge rgb, flow, and iDT results')


# parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('--rgb', type=str, default=None)
parser.add_argument('--flow', type=str, default=None)
parser.add_argument('--idt', type=str, default=None)
parser.add_argument('--weight' ,type=float, default=[1., 1.5], nargs="+")

args = parser.parse_args()

def load_score(file_path):
    score_info = np.load(file_path)
    return score_info['scores'], score_info['labels']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_acc(rgb_score, flow_score, labels):

    acc = 0.
    for rgb, flow, label in zip(rgb_score, flow_score, labels):
        
        # rgb = softmax(rgb[0][0][0])
        # flow = softmax(flow[0][0][0])

        rgb = rgb[0][0][0]
        flow = flow[0][0][0]

        final_pred = np.asarray(rgb)*args.weight[0] + np.asarray(flow)*args.weight[1]
        acc += 1 if np.argmax(final_pred) == label else 0

    print('Accuracy {:.02f}%'.format(np.mean(acc*1.0/labels.shape[0] * 100)))

    # np.argmax(np.mean(x[0], axis=0))
    # cf = confusion_matrix(video_labels, video_pred).astype(float)
    # cls_cnt = cf.sum(axis=1)
    # cls_hit = np.diag(cf)
    # cls_acc = cls_hit / cls_cnt
    # print(cls_acc)


if __name__ == '__main__':
    
    rgb_score, labels = load_score(args.rgb)
    flow_score, labels = load_score(args.flow)
    compute_acc(rgb_score, flow_score, labels)




