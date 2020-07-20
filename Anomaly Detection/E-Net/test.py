import torch
import torch.nn as nn
import anom_utils
import numpy as np 
import torchvision.transforms as transforms
import utils
import transforms as ext_transforms

def eval_ood_measure(conf, seg_label):

    out_label = seg_label == 11
    out_label = np.logical_or(out_label, seg_label == 11)
    in_scores  = - conf[np.logical_not(out_label)]
    out_scores = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None

class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        aurocs = {'max_logit':[],'msp':[],'backg':[]}
        auprs  = {'max_logit':[],'msp':[],'backg':[]}
        fprs   = {'max_logit':[],'msp':[],'backg':[]}
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            
            # =============================================================== #
            #       remove label 'car', 'pedestrian' and 'bicyclist'          #
            # =============================================================== #
            mask_label = (batch_data[1] == 8) | (batch_data[1] == 9) | (batch_data[1] == 10)
            batch_data[1][mask_label] = 11
            
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            
            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)
                
                # Loss computation
                loss = self.criterion(outputs, labels)
            
            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            
            # Keep track of evaluation the metric
            self.metric.add(outputs.detach(), labels.detach())
            if iteration_loss:
                print("[batch:{0:2d}] Iteration loss: {1:.4f}".format(step, loss.item()))

            # get other metric (cf Hendrycks)
            max_logit, _ = torch.max(outputs.data, 1)
            msp, _       = torch.max(nn.functional.softmax(outputs.data, 1), 1)
            backg        = outputs.data[:, 11]
            
            # evaluate performances 
            auroc, aupr, fpr = eval_ood_measure(max_logit.cpu(), labels.cpu())
            aurocs['max_logit'].append(auroc); auprs['max_logit'].append(aupr), fprs['max_logit'].append(fpr)
            auroc, aupr, fpr = eval_ood_measure(msp.cpu(), labels.cpu())
            aurocs['msp'].append(auroc); auprs['msp'].append(aupr), fprs['msp'].append(fpr)
            auroc, aupr, fpr = eval_ood_measure(backg.cpu(), labels.cpu())
            aurocs['backg'].append(auroc); auprs['backg'].append(aupr), fprs['backg'].append(fpr)                        

        return epoch_loss / len(self.data_loader), self.metric.value(), aurocs, auprs, fprs
    
    def print_results(self, loss, miou, class_encoding, iou, aurocs, auprs, fprs, args):
        
        # Print Loss and IoU
        print(">>>> Avg. loss: {:.4f}".format(loss))
        print(">>>> Avg. IoU : {:.4f}".format(miou))
        print(4*" "+22*"-")
        print(4*" "+"|{0:11}| {1:6} |".format("classes", "IoU"))
        print(4*" "+22*"-")
        for key, class_iou in zip(class_encoding.keys(), iou):
            print(4*" "+"|{0:11}| {1:.4f} |".format(key, class_iou))
        print(4*" "+22*"-")
        
        # print AUROC, AUPR and FPR
        print("\n>>>>      | mean(AUROC) | mean(AUPR) | mean(FPR) |")
        print(10*" "+40*"-")
        print("max_logit |    {0:.4f}   |   {1:.4f}   |   {2:.4f}  |".format(np.mean(aurocs['max_logit']), np.mean(auprs['max_logit']),np.mean(fprs['max_logit'])))
        print("msp       |    {0:.4f}   |   {1:.4f}   |   {2:.4f}  |".format(np.mean(aurocs['msp']), np.mean(auprs['msp']),np.mean(fprs['msp'])))
        print("backg     |    {0:.4f}   |   {1:.4f}   |   {2:.4f}  |".format(np.mean(aurocs['backg']), np.mean(auprs['backg']),np.mean(fprs['backg'])))
        print(10*" "+40*"-")
        
        # Display a batch of samples and labels
        if args.imshow_batch:
            print("\nA batch of predictions from the test set...")
            images, _ = iter(self.data_loader).next()
            images = images.to(args.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
            max_logit, max_index = torch.max(predictions.data, 1)
            msp, _       = torch.max(nn.functional.softmax(predictions.data, dim=1),dim=1)
            backg        = predictions.data[:, 11]
            label_to_rgb = transforms.Compose([ext_transforms.LongTensorToRGBPIL(class_encoding), transforms.ToTensor()])
            color_predictions = utils.batch_transform(max_index.cpu(), label_to_rgb)
            utils.imshow_metrics(images.data.cpu(), color_predictions, msp, max_logit, backg, args.batch_size)
