
package_path = '../input/unetmodelscript' # add unet script dataset

import sys

sys.path.append(package_path)

from model import Unet # import Unet model from the script
import os

import cv2

import pdb

import time

import warnings

import random

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

from torch.nn import functional as F

import torch.optim as optim

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset, sampler

from matplotlib import pyplot as plt

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)

from albumentations.pytorch import ToTensor

warnings.filterwarnings("ignore")

seed = 69

random.seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)

np.random.seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True

from sklearn import model_selection

stage =1 ## Change this value to 2 , to load the stage 1 model and refine with tversky loss
#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask2rle(img):

    '''

    img: numpy array, 1 -> mask, 0 -> background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def make_mask(row_id, df):

    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''

    fname = df.iloc[row_id].name

    labels = df.iloc[row_id][:4]

    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp

    # 4:class 1～4 (ch:0～3)



    for idx, label in enumerate(labels.values):

        if label is not np.nan:

            label = label.split(" ")

            positions = map(int, label[0::2])

            length = map(int, label[1::2])

            mask = np.zeros(256 * 1600, dtype=np.uint8)

            for pos, le in zip(positions, length):

                mask[pos:(pos + le)] = 1

            masks[:, :, idx] = mask.reshape(256, 1600, order='F')

    return fname, masks
class SteelDataset(Dataset):

    def __init__(self, df, data_folder, mean, std, phase):

        self.df = df

        self.root = data_folder

        self.mean = mean

        self.std = std

        self.phase = phase

        self.transforms = get_transforms(phase, mean, std)

        self.fnames = self.df.index.tolist()



    def __getitem__(self, idx):

        image_id, mask = make_mask(idx, self.df)

        image_path = os.path.join(self.root, "train_images",  image_id)

        img = cv2.imread(image_path)

        augmented = self.transforms(image=img, mask=mask)

        img = augmented['image']

        mask = augmented['mask'] # 1x256x1600x4

        mask = mask[0].permute(2, 0, 1) # 1x4x256x1600

        return img, mask



    def __len__(self):

        return len(self.fnames)





def get_transforms(phase, mean, std):

    list_transforms = []

    if phase == "train":

        list_transforms.extend(

            [

                HorizontalFlip(p=0.5), # only horizontal flip as of now

            ]

        )

    list_transforms.extend(

        [

            Normalize(mean=mean, std=std, p=1),

            ToTensor(),

        ]

    )

    list_trfms = Compose(list_transforms)

    return list_trfms



def provider(

    data_folder,

    train_df,

    val_df,

    phase,

    mean=None,

    std=None,

    batch_size=8,

    num_workers=4,

):

    '''Returns dataloader for the model training'''

   # train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)

    df = train_df if phase == "train" else val_df

    image_dataset = SteelDataset(df, data_folder, mean, std, phase)

    dataloader = DataLoader(

        image_dataset,

        batch_size=batch_size,

        num_workers=num_workers,

        pin_memory=True,

        shuffle=True,   

    )



    return dataloader

def predict(X, threshold):

    '''X is sigmoid output of the model'''

    X_p = np.copy(X)

    preds = (X_p > threshold).astype('uint8')

    return preds



def metric(probability, truth, threshold=0.5, reduction='none'):

    '''Calculates dice of positive and negative images seperately'''

    '''probability and truth must be torch tensors'''

    batch_size = len(truth)

    with torch.no_grad():

        probability = probability.view(batch_size, -1)

        truth = truth.view(batch_size, -1)

        assert(probability.shape == truth.shape)



        p = (probability > threshold).float()

        t = (truth > 0.5).float()



        t_sum = t.sum(-1)

        p_sum = p.sum(-1)

        neg_index = torch.nonzero(t_sum == 0)

        pos_index = torch.nonzero(t_sum >= 1)



        dice_neg = (p_sum == 0).float()

        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))



        dice_neg = dice_neg[neg_index]

        dice_pos = dice_pos[pos_index]

        dice = torch.cat([dice_pos, dice_neg])



        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)

        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)

        dice = dice.mean().item()



        num_neg = len(neg_index)

        num_pos = len(pos_index)



    return dice, dice_neg, dice_pos, num_neg, num_pos



class Meter:

    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, phase, epoch):

        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold

        self.base_dice_scores = []

        self.dice_neg_scores = []

        self.dice_pos_scores = []

        self.iou_scores = []



    def update(self, targets, outputs):

        probs = torch.sigmoid(outputs)

        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)

        self.base_dice_scores.append(dice)

        self.dice_pos_scores.append(dice_pos)

        self.dice_neg_scores.append(dice_neg)

        preds = predict(probs, self.base_threshold)

        iou = compute_iou_batch(preds, targets, classes=[1])

        self.iou_scores.append(iou)



    def get_metrics(self):

        dice = np.mean(self.base_dice_scores)

        dice_neg = np.mean(self.dice_neg_scores)

        dice_pos = np.mean(self.dice_pos_scores)

        dices = [dice, dice_neg, dice_pos]

        iou = np.nanmean(self.iou_scores)

        return dices, iou



def epoch_log(phase, epoch, epoch_loss, meter, start):

    '''logging the metrics at the end of an epoch'''

    dices, iou = meter.get_metrics()

    dice, dice_neg, dice_pos = dices

    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))

    return dice, iou



def compute_ious(pred, label, classes, ignore_index=255, only_present=True):

    '''computes iou for one ground truth mask and predicted mask'''

    pred[label == ignore_index] = 0

    ious = []

    for c in classes:

        label_c = label == c

        if only_present and np.sum(label_c) == 0:

            ious.append(np.nan)

            continue

        pred_c = pred == c

        intersection = np.logical_and(pred_c, label_c).sum()

        union = np.logical_or(pred_c, label_c).sum()

        if union != 0:

            ious.append(intersection / union)

    return ious if ious else [1]



def compute_iou_batch(outputs, labels, classes=None):

    '''computes mean iou for a batch of ground truth masks and predicted masks'''

    ious = []

    preds = np.copy(outputs) # copy is imp

    labels = np.array(labels) # tensor to np

    for pred, label in zip(preds, labels):

        ious.append(np.nanmean(compute_ious(pred, label, classes)))

    iou = np.nanmean(ious)

    return iou

import segmentation_models_pytorch as sem

import torchcontrib
from segmentation_models_pytorch.encoders import get_preprocessing_fn



preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
model = sem.FPN("resnet34", encoder_weights="imagenet", classes=4, activation=None)#
if stage ==2 : ## Load the pretrained model generated by stage 1 here .

    ckpt_path = "./input/stage1/model_fpn_stage_1_fold_1.pth"  #Model for FPN

    device = torch.device("cuda")

    model = sem.FPN("resnet34", encoder_weights="imagenet", classes=4, activation=None)

    model.to(device)

    #model.eval()

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(state["state_dict"])
model # a *deeper* look
#PyTorch

ALPHA = 0.5

BETA = 0.5

GAMMA = 1



class FocalTverskyLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):

        super(FocalTverskyLoss, self).__init__()



    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):

        

        #comment out if your model contains a sigmoid or equivalent activation layer

        inputs = F.sigmoid(inputs)       

        

        #flatten label and prediction tensors

        inputs = inputs.view(-1)

        targets = targets.view(-1)

        

        #True Positives, False Positives & False Negatives

        TP = (inputs * targets).sum()    

        FP = ((1-targets) * inputs).sum()

        FN = (targets * (1-inputs)).sum()

        

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  

        FocalTversky = (1 - Tversky)**gamma

                       

        return FocalTversky
class Trainer(object):

    '''This class takes care of training and validation of our model'''

    def __init__(self, model, train_df, val_df, fold):

        self.num_workers = 6

        self.batch_size = {"train": 4, "val": 4}

        self.accumulation_steps = 32 // self.batch_size['train']

        self.lr = 5e-4

        self.num_epochs = 2

        self.stage = stage

        #20

        self.best_loss = float("inf")

        self.phases = ["train", "val"]

        self.device = torch.device("cuda:0")

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self.net = model

        weight = [5,10,2,5] ## to handle class imbalance

        weight = torch.FloatTensor(weight).cuda()

        self.weight = torch.reshape(weight,(1,4,1,1))

        if stage ==1:

            self.criterion = torch.nn.BCEWithLogitsLoss(weight=self.weight, reduce=False)

        else:

            self.criterion = FocalTverskyLoss(weight=self.weight)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)

        self.net = self.net.to(self.device)

        self.fold = fold

        cudnn.benchmark = True

        self.dataloaders = {

            phase: provider(

                data_folder=data_folder,

                train_df=train_df,

                val_df=val_df,

                phase=phase,

                mean=(0.485, 0.456, 0.406),

                std=(0.229, 0.224, 0.225),

                batch_size=self.batch_size[phase],

                num_workers=self.num_workers,

            )

            for phase in self.phases

        }

        self.losses = {phase: [] for phase in self.phases}

        self.iou_scores = {phase: [] for phase in self.phases}

        self.dice_scores = {phase: [] for phase in self.phases}

        

    def forward(self, images, targets):

        images = images.to(self.device)

        masks = targets.to(self.device)

        outputs = self.net(images)

        loss = self.criterion(outputs, masks)

        loss = loss.mean()

        return loss, outputs



    def iterate(self, epoch, phase):

        meter = Meter(phase, epoch)

        start = time.strftime("%H:%M:%S")

        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        batch_size = self.batch_size[phase]

        self.net.train(phase == "train")

        dataloader = self.dataloaders[phase]

        running_loss = 0.0

        total_batches = len(dataloader)

        tk0 = tqdm(dataloader, total=total_batches)

        self.optimizer.zero_grad()

        for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm

            images, targets = batch

            loss, outputs = self.forward(images, targets)

            loss = loss / self.accumulation_steps

            if phase == "train":

                loss.backward()

                if (itr + 1 ) % self.accumulation_steps == 0:

                    self.optimizer.step()

                    self.optimizer.zero_grad()

            running_loss += loss.item()

            outputs = outputs.detach().cpu()

            meter.update(targets, outputs)

            tk0.set_postfix(loss=(running_loss / ((itr + 1))))

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches

        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)

        self.losses[phase].append(epoch_loss)

        self.dice_scores[phase].append(dice)

        self.iou_scores[phase].append(iou)

        torch.cuda.empty_cache()

        return epoch_loss



    def start(self):

        for epoch in range(self.num_epochs):

            self.iterate(epoch, "train")

            state = {

                "epoch": epoch,

                "best_loss": self.best_loss,

                "state_dict": self.net.state_dict(),

                "optimizer": self.optimizer.state_dict(),

            }

            with torch.no_grad():

                val_loss = self.iterate(epoch, "val")

                self.scheduler.step(val_loss)

            if val_loss < self.best_loss:

                print("******** New optimal found, saving state ********")

                state["best_loss"] = self.best_loss = val_loss

                torch.save(state, f"./model_fpn_stage_{self.stage}_fold_{self.fold}.pth")

            print()

sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'

train_df_path = '../input/severstal-steel-defect-detection/train.csv'

data_folder = "../input/severstal-steel-defect-detection/"

test_data_folder = "../input/severstal-steel-defect-detection/test_images"
def plot(scores, name):

    plt.figure(figsize=(15,5))

    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')

    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')

    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');

    plt.legend(); 

    plt.show()

df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

# https://www.kaggle.com/amanooo/defect-detection-starter-u-net

df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))

df['ClassId'] = df['ClassId'].astype(int)

df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')

df['defects'] = df.count(axis=1)



fold = model_selection.StratifiedKFold(n_splits=4, shuffle=False,)



for index, (train_idx, val_idx) in enumerate(fold.split(df, df['defects'])):

    train = df.iloc[train_idx]

    val = df.iloc[val_idx]

    print('fold', index, train.shape, val.shape)

    model_trainer = Trainer(model, train, val, index)

    model_trainer.start()

    # PLOT TRAINING

    losses = model_trainer.losses

    dice_scores = model_trainer.dice_scores # overall dice

    iou_scores = model_trainer.iou_scores

    plot(losses, "BCE loss")

    plot(dice_scores, "Dice score")

    plot(iou_scores, "IoU score")
# PLOT TRAINING

# losses = model_trainer.losses

# dice_scores = model_trainer.dice_scores # overall dice

# iou_scores = model_trainer.iou_scores



# def plot(scores, name):

#     plt.figure(figsize=(15,5))

#     plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')

#     plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')

#     plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');

#     plt.legend(); 

#     plt.show()



# plot(losses, "BCE loss")

# plot(dice_scores, "Dice score")

# plot(iou_scores, "IoU score")