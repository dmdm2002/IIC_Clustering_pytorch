import tqdm
import torch
from torch.nn import functional as F
from Utils.Option import Param
from Utils.Visualizer import Visualizer


class TrainOneEpoch(Param):
    def __init__(self):
        super(TrainOneEpoch, self).__init__()

    def run_head_epoch(self, model, head, dataloader, loss_fn, optimizer, disp):
        """
        :param model: Training Model
        :param head: main head(B) or aux head(A)
        :param dataloader: Call Dataset
        :param loss_fn: IID Loss
        :param optimizer: Adam
        :param disp: Accuracy and loss calculate
        :return: Trained model
        """
        model = model
        for head_ep in range(self.HeadEpoch):
            for idx, (item, item_aug, _, name) in enumerate(tqdm.tqdm(dataloader, desc=f'HEAD({head}) TRAINING EPOCH [{head_ep}/{self.HeadEpoch}]')):
                avg_loss_batch = None
                avg_loss_no_lamb_batch = None

                item = item.to(self.DEVICE)
                item_aug = item_aug.to(self.DEVICE)

                output = model(item)
                output_aug = model(item_aug)

                for i in range(self.NumSubHeads):
                    loss_batch, loss_no_lamb_batch = loss_fn.IID_loss(output[i], output_aug[i])

                    if avg_loss_batch is None:
                        avg_loss_batch = loss_batch
                        avg_loss_no_lamb_batch = loss_no_lamb_batch
                    else:
                        avg_loss_batch += loss_batch
                        avg_loss_no_lamb_batch += loss_no_lamb_batch

                avg_loss_batch /= self.NumSubHeads
                avg_loss_no_lamb_batch /= self.NumSubHeads

                if head == 'B':
                    scores = [[0, avg_loss_batch.item()], [1, avg_loss_no_lamb_batch.item()]]
                else:
                    scores = [[2, avg_loss_batch.item()], [3, avg_loss_no_lamb_batch.item()]]

                disp.record(scores, select_index=True)

                avg_loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model

    def __call__(self, model, head, dataloader, loss_fn, optimizer, disp):
        return self.run_head_epoch(model, head, dataloader, loss_fn, optimizer, disp)


class EvalOneEpoch(Param):
    def __init__(self):
        super(EvalOneEpoch, self).__init__()

    def run_valid(self, model, head, dataloader, ep, disp):
        """
        :param model: Validation model
        :param head: main head(B) or aux head(A)
        :param dataloader: Call Dataset
        :param ep: Now epoch
        :param disp: Accuracy and loss calculate
        :return: Not
        """
        model = model
        with torch.no_grad():
            for idx, (item, _, label, name) in enumerate(tqdm.tqdm(dataloader, desc=f'HEAD({head}) VALIDATION EPOCH [{ep}/{self.HeadEpoch}]')):
                item = item.to(self.DEVICE)
                label = label.to(self.DEVICE)

                output = model(item, head=head)
                for i in range(self.NumSubHeads):
                    assignments = output[i].argmax(dim=1)
                    predict = 0
                    if assignments == label:
                        predict = 1

                    disp.record([predict])

    def run_valid_with_visualization(self, model, head, dataloader, ep, disp):
        """
        :param model: Validation model
        :param head: main head(B) or aux head(A)
        :param dataloader: Call Dataset
        :param ep: Now epoch
        :param disp: Accuracy and loss calculate
        :return: Not
        """
        model = model
        deepfeatures = []
        actuals = []
        with torch.no_grad():
            for idx, (item, _, label, name) in enumerate(
                    tqdm.tqdm(dataloader, desc=f'HEAD({head}) VALIDATION EPOCH [{ep}/{self.HeadEpoch}]')):
                item = item.to(self.DEVICE)
                label = label.to(self.DEVICE)

                deepfeature = model(item, head=head, kmeans_use_features=True)
                output = model(item, head=head)
                for i in range(self.NumSubHeads):
                    deepfeature = deepfeature[i]
                    assignments = output[i].argmax(dim=1)
                    predict = 0
                    predict += (assignments == label).type(torch.float).sum().item()

                    disp.record([predict])

                    deepfeatures += deepfeature.cpu().numpy().tolist()
                    actuals += label.cpu().numpy().tolist()

        Visualizer()(deepfeatures, actuals, ep)

    def __call__(self, model, head, dataloader, ep, disp, visualization=False):
        if visualization:
            return self.run_valid_with_visualization(model, head, dataloader, ep, disp)
        else:
            return self.run_valid(model, head, dataloader, ep, disp)
