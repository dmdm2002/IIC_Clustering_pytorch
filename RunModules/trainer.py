import os
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from Utils.Option import Param
from Utils.Displayer import LossAccDisplayer
from Utils.Functinos import CkpHandler, TensorboardHandler, TransformBuilder
from Utils.iic_loss import CalLoss
from Utils.ErrorLogger import error_logger
from Utils.CustomDataset import Datasets
from Model.cluster.ClusterModel2 import ClusterNetHead

from RunModules.RunEpoch import TrainOneEpoch, EvalOneEpoch


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Train(Param):
    def __init__(self):
        super(Train, self).__init__()
        # Displayer
        self.tr_disp = LossAccDisplayer(self.tr_name_type_list)
        self.valid_disp = LossAccDisplayer(self.valid_name_type_list)

        # Call Functions
        self.ckp_handler = CkpHandler()
        self.transform_builder = TransformBuilder()
        self.tensorboard_handler = TensorboardHandler()

        # make output folders
        os.makedirs(self.OUTPUT_LOG, exist_ok=True)
        os.makedirs(self.OUTPUT_CKP, exist_ok=True)

        error_path = f'{self.OUTPUT_LOG}/ErrorLog.csv'
        if not os.path.isfile(error_path):
            df = pd.DataFrame(columns=['Folder', 'ImageName', 'Class'])
            df.to_csv(error_path, index=False)

    def run(self):
        print('--------------------------------------')
        print(f'[RunType] : Training!!')
        print(f'[Device] : {self.DEVICE}!!')
        print('--------------------------------------')

        # model = IICModel(n_classes=self.NumClasses, aux_classes=self.AuxClass)
        model = ClusterNetHead(label_dim=self.NumClasses, aux_output_dim=self.AuxClass, num_sub_heads=self.NumSubHeads)
        model, epoch = self.ckp_handler.load_ckp(model, imagenet=True)
        model = model.to(self.DEVICE)

        tr_transform = self.transform_builder.set_train_transform(do_aug=False)
        if self.AUG:
            tr_aug_tranform = self.transform_builder.set_train_transform(do_aug=True)
            tr_dataset = Datasets(self.DATASET_PATH, run_type='train', transform=tr_transform,
                                  aug_transform=tr_aug_tranform)
        else:
            tr_dataset = Datasets(self.DATASET_PATH, run_type='train', transform=tr_transform)

        val_transform = self.transform_builder.set_valid_test_transform()
        val_dataset = Datasets(self.DATASET_PATH, run_type='test', transform=val_transform, aug_transform=None)

        loss_fn = CalLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.LR)

        for ep in range(epoch, self.EPOCH):
            tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCHSZ, shuffle=False)

            model.train()
            print('--------------------------------------')
            print(f'[Now Loop] : Training Classifier!!')
            print(f'[NOW Training EPOCH] : {ep}/{self.EPOCH}')
            print('--------------------------------------')
            for head in self.HeadList:
                model = TrainOneEpoch()(model, head, tr_loader, loss_fn, optimizer, self.tr_disp)

            self.tr_disp.get_avg_losses(length=len(tr_dataset))
            self.tr_disp.reset()

            model.eval()
            print('--------------------------------------')
            print(f'[Now Loop] : Valid Classifier!!')
            print(f'[NOW VALIDATION EPOCH] : {ep}/{self.EPOCH}')
            print('--------------------------------------')
            EvalOneEpoch()(model, "B", val_loader, ep, self.valid_disp, visualization=self.Visualization)
            self.valid_disp.get_avg_losses(length=len(val_dataset))
            self.valid_disp.reset()

            self.ckp_handler.save_ckp(model=model, optimizer=optimizer, scheduler=None, epoch=ep)
