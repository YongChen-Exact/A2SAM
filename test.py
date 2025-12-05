import argparse
import os

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import datasets
import models
import utils

from torchvision import transforms
from evaluation.EvalMetric import cal_metrics
torch.cuda.set_device(2)


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)




def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):

    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp, batch['unc'], batch['lgt']))
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()



def predict(loader, model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model.eval()
    pbar = tqdm(loader, leave=False, desc='val')
    pred_all=[]
    gt_all=[]
    names=[]
    for batch in pbar:
        for k in ['inp', 'gt', 'unc','lgt']:
            if k in batch:
                batch[k] = batch[k].cuda()
        # for k, v in batch.items():
        #     batch[k] = v.cuda()
        names.append(batch['name'][0])
        inp = batch['inp']
        pred = torch.sigmoid(model.infer(inp, batch['unc'], batch['lgt']))[0][0]
        label=batch['gt'][0][0]

        pred1 = pred
        label, pred = label.cpu().data.numpy() * 255, pred.cpu().data.numpy() * 255

        pred = (pred > 128).astype(np.uint8)
        label = (label > 128).astype(np.uint8)
        pred_all.append(pred)
        gt_all.append(label)

        pred1 = ((pred1>0.5).cpu().data.numpy() * 255).astype(np.uint8)
        pred_img = Image.fromarray(pred1, mode='L')
        pred_img.save(os.path.join(path, batch['name'][0]))

    pred_all = np.stack(pred_all)
    gt_all = np.stack(gt_all)
    cal_metrics(path, names, pred_all, gt_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--save_path')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']


    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=1,
                        num_workers=8)
    config['model']["args"]['encoder_mode']["tuning_mode"] = None
    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=False)
    predict(loader, model, args.save_path)
