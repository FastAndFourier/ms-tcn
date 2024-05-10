#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], dtype=float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():

    data_path = "/media/isir/PHD/code/data_processing/Xml2Py/data/grasping"

    parser = argparse.ArgumentParser()
    parser.add_argument('--id')
    args = parser.parse_args()


    classes = "tri"
    sr = 100
    duration = 30
    modality = "both"


    data_fname = f'{classes}_{sr}_{duration}_{0}_{modality}'

    X = np.load(f"{data_path}/data_{data_fname}.npy")
    y = np.load(f"{data_path}/label_{data_fname}.npy")
    split = np.load(f"{data_path}/split_{data_fname}.npy", allow_pickle=True)[()]


    X = X[split['test']]
    y = y[split['test']]

    
    model = torch.load(f"models/ms_tcn/best_model_{args.id}.pt").cuda()
    model.eval()
    X_tensor = torch.from_numpy(X).float().cuda()
    mask = torch.ones_like(X_tensor).cuda()
    pred = torch.argmax(model(X_tensor, mask)[-1], axis=1).cpu().numpy()
    y = np.argmax(y, axis=1)


    for i in range(len(y)):
        plt.plot(y[i], label="gt")
        plt.plot(pred[i], label="pred")
        plt.legend()
        plt.show()    


    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for w in range(len(y)):

        for i in range(len(y[w])):
            total += 1
            if y[w,i] == pred[w,i]:
                correct += 1
        
        edit += edit_score(pred[w], y[w])

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred[w], y[w], overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        
    print("Acc: %.4f" % (100*float(correct)/total))
    print('Edit: %.4f' % ((1.0*edit)/len(y)))

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))


if __name__ == '__main__':
    main()
