
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import os

from scipy import stats

from tqdm import tqdm

from collections import OrderedDict , namedtuple

from torch.optim import lr_scheduler

import joblib

import logging 

import transformers

from transformers import AdamW , get_linear_schedule_with_warmup, get_constant_schedule

import sys

from sklearn import metrics , model_selection



import warnings

import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils



warnings.filterwarnings('ignore')

class AverageMeter:

    """

    Computes and stores the average and current value

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
class BERTModel(nn.Module):

    def __init__(self , bert_path):

        super(BERTModel , self).__init__()

        self.bert_path = bert_path

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear ( 768*2 , 1 )

        

    def forward(self , ids , mask , token_type_ids):

        o1 , o2 = self.bert ( ids , attention_mask = mask , token_type_ids = token_type_ids)

        avgpool = torch.mean(o1 , 1)

        maxpool, _ = torch.max(o1 ,1)

        cat = torch.cat((avgpool,maxpool) , 1 )

        bo = self.bert_drop(cat)

        p2 = self.out(bo)

        return p2
class BERTDataset:

    def __init__(self, comment_text, targets, tokenizer, max_length):

        self.comment_text = comment_text

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.targets = targets



    def __len__(self):

        return len(self.comment_text)



    def __getitem__(self, item):

        comment_text = str(self.comment_text[item])

        comment_text = " ".join(comment_text.split())



        inputs = self.tokenizer.encode_plus(

            comment_text,

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]

        

        padding_length = self.max_length - len(ids)

        

        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'targets': torch.tensor(self.targets[item], dtype=torch.float)

        }
mx = BERTModel(bert_path="../input/bert-base-multilingual-uncased/")

df_train1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"]).fillna("none")

df_train2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"]).fillna("none")

df_train_full = pd.concat([df_train1, df_train2], axis=0).reset_index(drop=True)

df_train = df_train_full.sample(frac=1).reset_index(drop=True).head(200000)



df_valid = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
def loss_fn(outputs ,targets):

    return nn.BCEWithLogitsLoss()(outputs , targets.view(-1,1))



def trainloop(data_loader , model , optimizer ,device , scheduler = None):

    model.train()

    for bi , d in enumerate(data_loader):

        ids = d['ids']

        targets = d['targets']

        mask = d['mask']

        token_type_ids = d['token_type_ids']

        

        ids=  ids.to(device , dtype = torch.long)

        targets=  targets.to(device , dtype = torch.float)

        mask=  mask.to(device , dtype = torch.long)

        token_type_ids=  token_type_ids.to(device , dtype = torch.long)

        

        optimizer.zero_grad()

        outputs = model(ids=ids,  mask=mask , token_type_ids = token_type_ids)

        

        loss=loss_fn(outputs ,targets)

        if bi%10 ==0:

            xm.master_print(f' Batch Index : {bi}  ||  Loss : {loss}')

        

                            

               

        loss.backward()

        xm.optimizer_step(optimizer)

        if scheduler is not None:

            scheduler.step()

            

            

def evalloop(data_loader , model , device):

    

    model.eval()

    fin_targets = []

    fin_outputs = []

    for bi, d in enumerate(data_loader):

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]

        targets = d["targets"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)



        outputs = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )



        targets_np = targets.cpu().detach().numpy().tolist()

        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_targets.extend(targets_np)

        fin_outputs.extend(outputs_np)

                



    return fin_outputs, fin_targets       

 
def _run():

    

    MAX_LEN = 192

    TRAIN_BATCH_SIZE = 64

    VAL_BATCH_SIZE = 64

    EPOCHS = 5

    

    tokenizer = transformers.BertTokenizer.from_pretrained('../input/bert-base-multilingual-uncased/' , do_lower_case = True)

    

    train_targets = df_train.toxic.values

    valid_targets = df_valid.toxic.values

    

    train_dataset = BERTDataset(comment_text = df_train.comment_text.values,

                                targets = train_targets,

                                tokenizer= tokenizer,

                                max_length = MAX_LEN

                               )

    

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,

                                                                    num_replicas = xm.xrt_world_size(),

                                                                    rank = xm.get_ordinal(),

                                                                    shuffle= True)

    

    train_dataloader = torch.utils.data.DataLoader(train_dataset,

                                                   batch_size=  TRAIN_BATCH_SIZE,

                                                   sampler = train_sampler,

                                                   drop_last = True,

                                                   num_workers = 4)

    

    

    val_dataset = BERTDataset(comment_text = df_valid.comment_text.values,

                                targets = valid_targets,

                                tokenizer= tokenizer,

                                max_length = MAX_LEN

                               )

    

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,

                                                                    num_replicas = xm.xrt_world_size(),

                                                                    rank = xm.get_ordinal(),

                                                                    shuffle= False)

    

    val_dataloader = torch.utils.data.DataLoader(val_dataset,

                                                   batch_size=  VAL_BATCH_SIZE,

                                                   sampler = val_sampler,

                                                   drop_last = False,

                                                   num_workers = 4)

    

    device = xm.xla_device()

    model = mx.to(device)

    

    

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias' , 'LayerNorm.bias' , 'LayerNorm.weight']

    optimizer_grouped_params = [

        {'params' : [p for n , p in param_optimizer if not any (nd in n for nd in no_decay)] , 'weight_decay' : 0.001},

        {'params' : [p for n , p in param_optimizer if any (nd in n for nd in no_decay)] , 'weight_decay' : 0.0}

       ]

    

    lr = 0.4 * 1e-5 * xm.xrt_world_size()

    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS )

    xm.master_print(f' NUM_TRAIN_STEPS : {num_train_steps} || XRT_WORLD_SIZE : { xm.xrt_world_size()}')

    

    

    optimizer = AdamW(optimizer_grouped_params,lr=lr)

    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0 , num_training_steps = num_train_steps)

    

    

    for epoch in range(EPOCHS):

        para_loader = pl.ParallelLoader(train_dataloader , [device])

        trainloop(para_loader.per_device_loader(device) , model, optimizer,  device,  scheduler=scheduler)

        para_loader = pl.ParallelLoader(val_dataloader , [device])

        o,t= evalloop(para_loader.per_device_loader(device) , model , device )

        xm.save(model.state_dict ,'bertmodel.bin')

        auc = metrics.roc_auc_score(np.array(t) >= 0.5 , o )

        xm.master_print(f'ROC-AUC Score : {auc}')

        

                        
def _multiprocessing_train(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = _run()

    

    
FLAGS = {}

xmp.spawn(_multiprocessing_train , args= (FLAGS,), nprocs = 8 , start_method = 'fork' )