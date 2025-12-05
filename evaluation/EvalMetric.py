import cv2, glob
import numpy as np
from skimage import morphology
from skimage.morphology import skeletonize
from medpy import metric
import os
import pandas as pd
np.set_printoptions(threshold=np.inf)
import csv
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-10)


def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)


def clDice_metric(v_p, v_l):
    tprec = cl_score(v_p, skeletonize(v_l))
    tsens = cl_score(v_l, skeletonize(v_p))
    return 2 * tprec * tsens / (tprec + tsens)


def calculate_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0


def calculate_assd(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0  and gt.sum() > 0:
        dice = metric.binary.assd(pred, gt)
        return dice
    else:
        return 0


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def cal_metrics(path, names, preds, gts):
    TP = FPN = 0
    Dice_ = []
    hd95 = []
    Jaccard_ = []
    clDice_ = []
    CAL_ = []
    precision_ = []
    recall_ = []
    assd = []

    preds = np.array(preds, dtype=np.uint8)
    gts = np.array(gts, dtype=np.uint8)

    for idx in range(preds.shape[0]):
        img = preds[idx]
        lab = gts[idx]

        if np.max(lab) == 255:
            img[img < 255] = 0
            lab[lab < 255] = 0
        elif np.max(lab) == 1:
            img[img < 1] = 0
            lab[lab < 1] = 0

        # Calculate Dice + Glob Jaccard
        TP = TP + np.sum(img * lab)
        FPN = FPN + np.sum(img) + np.sum(lab)
        single_I = np.sum(img * lab)
        single_U = np.sum(img) + np.sum(lab) - single_I
        Dice_.append(2 * TP / (FPN))
        Jaccard_.append(single_I / single_U)

        hd95.append(hd(img, lab))

        # Calculate clDice
        clDice_.append(clDice_metric(img, lab))

        precision_.append(metric.binary.precision(img, lab))
        recall_.append(metric.binary.recall(img, lab))
        assd.append(calculate_assd(img, lab))

        # Calculate connectivity
        ccs, _ = cv2.connectedComponents(img)
        ccsg, _ = cv2.connectedComponents(lab)
        numSg = np.count_nonzero(lab)
        epsilon = 1e-10  # 一个非常小的数
        C = 1 - min(abs(ccsg - ccs) / (numSg + epsilon), 1)

        # Calculate area
        kernel = np.array([[0, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 0, 0]], dtype=np.uint8)
        SDil = cv2.dilate(img, kernel)
        SgDil = cv2.dilate(lab, kernel)
        Anum = (SDil & lab) | (img & SgDil)
        Aden = img | lab
        A = np.count_nonzero(Anum) / np.count_nonzero(Aden)

        # Calculate length
        SSkel = morphology.skeletonize(img).astype(np.uint8)
        SgSkel = morphology.skeletonize(lab).astype(np.uint8)
        SDil = cv2.dilate(SSkel, kernel)
        SgDil = cv2.dilate(SgSkel, kernel)
        lnum = (SSkel & SgDil) | (SDil & SgSkel)
        lden = SSkel | SgSkel
        L = np.count_nonzero(lnum) / np.count_nonzero(lden)

        CAL = C * A * L
        CAL_.append(CAL)

    csv_avg_path = os.path.join(path, path.split("/")[-2] + '_' + path.split("/")[-1] +"_avg" + ".csv")
    csv_per_path = os.path.join(path, path.split("/")[-2] + '_' + path.split("/")[-1] +"_per" + ".csv")
    Dice = [round(x * 100,2) for x in Dice_]
    Jaccard = [round(x * 100, 2) for x in Jaccard_]
    precision = [round(x * 100, 2) for x in precision_]
    recall = [round(x * 100, 2) for x in recall_]
    clDice = [round(x * 100, 2) for x in clDice_]
    CAL = [round(x * 100, 2) for x in CAL_]
    hd95 = [round(x, 2) for x in hd95]
    assd = [round(x, 2) for x in assd]
    df = pd.DataFrame({
        "Name": names,
        "Dice": Dice,
        "Jac": Jaccard,
        "precision":precision,
        "recall":recall,
        "clDice": clDice,
        "CAL": CAL,
        "hd95": hd95,
        "assd": assd,
    })
    df.to_csv(csv_per_path, index=False)

    metrics = []
    stds=[]
    metrics.append(round(np.mean(Dice_) * 100, 2))
    metrics.append(round(np.mean(Jaccard_) * 100, 2))
    metrics.append(round(np.mean(precision_) * 100, 2))
    metrics.append(round(np.mean(recall_) * 100, 2))
    metrics.append(round(np.mean(clDice_) * 100, 2))
    metrics.append(round(np.mean(CAL_) * 100, 2))
    metrics.append(round(np.mean(hd95), 2))
    metrics.append(round(np.mean(assd), 2))

    stds.append(round(np.std(Dice_) * 100, 2))
    stds.append(round(np.std(Jaccard_) * 100, 2))
    stds.append(round(np.std(precision_) * 100, 2))
    stds.append(round(np.std(recall_) * 100, 2))
    stds.append(round(np.std(clDice_) * 100, 2))
    stds.append(round(np.std(CAL_) * 100, 2))
    stds.append(round(np.std(hd95), 2))
    stds.append(round(np.std(assd), 2))

    combined = [f"{m}±{s}" for m, s in zip(metrics, stds)]

    with open(csv_avg_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(combined)

    print("\nDice: %.2f±%.2f" % (np.mean(Dice_) * 100, np.std(Dice_) * 100))
    print("Jac: %.2f±%.2f" % (np.mean(Jaccard_) * 100, np.std(Jaccard_) * 100))
    print("precision: %.2f±%.2f" % (np.mean(precision_) * 100, np.std(precision_) * 100))
    print("recall: %.2f±%.2f" % (np.mean(recall_) * 100, np.std(recall_) * 100))
    print("clDice: %.2f±%.2f" % (np.mean(clDice_) * 100, np.std(clDice_) * 100))
    print("CAL: %.2f±%.2f" % (np.mean(CAL_) * 100, np.std(CAL_) * 100))
    print("hd95: %.2f±%.2f" % (np.mean(hd95), np.std(hd95)))
    print("assd: %.2f±%.2f" % (np.mean(assd), np.std(assd)))
