
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch.nn as nn

import torch

import transformers

from tqdm import tqdm

from sklearn import model_selection

from sklearn import metrics

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.distributed.parallel_loader as pl
MAX_LEN = 192

TRAIN_BATCH_SIZE = 64

VALID_BATCH_SIZE = 64

EPOCHS = 2

BERT_PATH = "../input/bert-base-multilingual-uncased//"

MODEL_PATH = "model.bin"

TOKENIZER = transformers.BertTokenizer.from_pretrained(

    BERT_PATH,

    do_lower_case=True

)
class AverageMeter:

    def __init__(self):

        self.reset()

        

    def reset(self):

        self.val=0

        self.avg=0

        self.sum=0

        self.count=0

        

    def update(self,val,n=1):

        self.val=val

        self.sum+=val*n

        self.count+=n

        self.avg=self.sum/self.count

        
class BERTBaseUncased(nn.Module):

    def __init__(self):

        super(BERTBaseUncased, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768*2, 1)

    

    def forward(self, ids, mask, token_type_ids):

        o1, _ = self.bert(

            ids, 

            attention_mask=mask,

            token_type_ids=token_type_ids

        )



        mean_pooling = torch.mean(o1,1)

        max_pooling,_=torch.max(o1,1)

        cat=torch.cat((mean_pooling,max_pooling),1)



        bo = self.bert_drop(cat)

        output = self.out(bo)

        return output
class BERTDataset:

    def __init__(self, comment_text, target):

        self.comment_text = comment_text

        self.target = target

        self.tokenizer = TOKENIZER

        self.max_len = MAX_LEN

    

    def __len__(self):

        return len(self.comment_text)

    

    def __getitem__(self, item):

        comment_text = str(self.comment_text[item])

        comment_text = " ".join(comment_text.split())



        inputs = self.tokenizer.encode_plus(

            comment_text,

            None,

            add_special_tokens=True,

            max_length=self.max_len,

            pad_to_max_length=True

        )



        ids = inputs["input_ids"]

        mask = inputs["attention_mask"]

        token_type_ids = inputs["token_type_ids"]



        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'targets': torch.tensor(self.target[item], dtype=torch.float)

        }
def loss_fn(outputs, targets):

    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))





def train_fn(data_loader, model, optimizer, device, scheduler):

    model.train()



    for bi, d in enumerate(data_loader):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets = d["targets"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)



        optimizer.zero_grad()

        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        loss = loss_fn(outputs, targets)

        loss.backward()

        xm.optimizer_step(optimizer)

        scheduler.step()

        if bi%10==0:

            xm.master_print(f"bi={bi}, loss={loss.item()}")



def eval_fn(data_loader, model, device):

    model.eval()

    fin_targets = []

    fin_outputs = []

    for bi, d in enumerate(data_loader):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)

        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

           )

        fin_targets.extend(targets.cpu().detach().numpy().tolist())

        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets
def run():

    df1=pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv",usecols=["comment_text","toxic"])

    df2=pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv",usecols=["comment_text","toxic"])



    df_train_full=pd.concat([df1,df2],axis=0).reset_index(drop=True)



    df_train=df_train_full.sample(frac=1).reset_index(drop=True).head(20000)

    

    df_valid=pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/validation.csv")





    train_dataset = BERTDataset(

        comment_text=df_train.comment_text.values,

        target=df_train.toxic.values,

        #tokenizer=TOKENIZER,

        #max_length=MAX_LEN

    )

    

    train_sampler=torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        num_workers=4,

        sampler=train_sampler,

        drop_last=True

    )



    valid_dataset = BERTDataset(

        comment_text=df_valid.comment_text.values,

        target=df_valid.toxic.values,

        #tokenizer=TOKENIZER,

        #max_length=MAX_LEN

    )

    

    valid_sampler=torch.utils.data.distributed.DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=VALID_BATCH_SIZE,

        num_workers=1

    )



    device = xm.xla_device()

    model = BERTBaseUncased()

    model.to(device)

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]



    #num_train_steps = int(len(df_train) /TRAIN_BATCH_SIZE/xm.xrt_world_size() *EPOCHS)

    num_train_steps = int(len(train_dataset) /TRAIN_BATCH_SIZE/xm.xrt_world_size() *EPOCHS)

    lr=3e-5 * xm.xrt_world_size()

    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

    optimizer = AdamW(optimizer_parameters, lr=lr)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )





    best_accuracy = 0

    for epoch in range(EPOCHS):

        para_loader = pl.ParallelLoader(train_data_loader, [device])

        train_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler)

        

        para_loader=pl.ParallelLoader(train_data_loader,[device])

        outputs, targets = eval_fn(para_loader.per_device_loader(device), model, device)

        

        xm.save(model.state_dict(), "model.bin")

        auc = metrics.roc_auc_score(np.array(targets) >= 0.5, outputs)

        xm.master_print(f'AUC = {auc}')



def _multiprocessing_function(rank,flags):

    torch.set_default_tensor_type("torch.FloatTensor")

    a=run()

    

xmp.spawn(_multiprocessing_function,args=({}, ),nprocs=1,start_method='fork')
import os

os.chdir(r'/kaggle/working')
from IPython.display import FileLink

FileLink(r'model.bin')