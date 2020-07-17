# System libs
import time
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
# Our libs
from utils import AverageMeter, accuracy, intersectionAndUnion
from lib.nn import async_copy_to
from lib.utils import as_numpy
from tqdm import tqdm

import anom_utils


def eval_ood_measure(conf, seg_label, cfg, mask=None):
    out_labels = cfg.OOD.out_label
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores  = - conf[np.logical_not(out_label)]
    out_scores = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    aurocs, auprs, fprs = [], [], []

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data       = batch_data[0]
        seg_label        = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores  = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores  = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores     = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            tmp_scores = scores
            if cfg.OOD.exclude_back:
                tmp_scores = tmp_scores[:,1:]


            mask    = None
            _, pred = torch.max(scores, dim=1)
            pred    = as_numpy(pred.squeeze(0).cpu())

            #for evaluating MSP
            if cfg.OOD.ood == "msp":
                conf, _  = torch.max(nn.functional.softmax(tmp_scores, dim=1),dim=1)
                conf     = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "maxlogit":
                conf, _  = torch.max(tmp_scores,dim=1)
                conf     = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "background":
                conf = tmp_scores[:, 0]
                conf = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "crf":
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
                ch,h,w = scores.squeeze(0).size()
                d      = dcrf.DenseCRF2D(h, w, ch)  # width, height, nlabels
                tmp_scores = as_numpy(nn.functional.softmax(tmp_scores, dim=1).squeeze(0))
                tmp_scores = as_numpy(tmp_scores)
                U = unary_from_softmax(tmp_scores)
                d.setUnaryEnergy(U)

                pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=13, img=tmp_scores, chdim=0)
                d.addPairwiseEnergy(pairwise_energy, compat=10)
                # Run inference for 100 iterations
                Q_unary = d.inference(100)
                # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                map_soln_unary = np.argmax(Q_unary, axis=0)

                # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                map_soln_unary = map_soln_unary.reshape((h,w))
                conf           = np.max(Q_unary, axis=0).reshape((h,w))
            elif cfg.OOD.ood == "crf-gauss":
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
                ch,h,w     = scores.squeeze(0).size()
                d          = dcrf.DenseCRF2D(h, w, ch)  # width, height, nlabels
                tmp_scores = as_numpy(nn.functional.softmax(tmp_scores, dim=1).squeeze(0))
                tmp_scores = as_numpy(tmp_scores)
                U          = unary_from_softmax(tmp_scores)
                d.setUnaryEnergy(U)
                d.addPairwiseGaussian(sxy=3, compat=3)  # `compat` is the "strength" of this potential.

                # Run inference for 100 iterations
                Q_unary = d.inference(100)
                # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                map_soln_unary = np.argmax(Q_unary, axis=0)

                # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                map_soln_unary = map_soln_unary.reshape((h,w))
                conf           = np.max(Q_unary, axis=0).reshape((h,w))
                
            res = eval_ood_measure(conf, seg_label, cfg, mask=mask)
            
            if res is not None:
                auroc, aupr, fpr = res
                aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)
            else:
                pass


        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'.format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    print("mean auroc = ", np.mean(aurocs), "mean aupr = ", np.mean(auprs), " mean fpr = ", np.mean(fprs))

