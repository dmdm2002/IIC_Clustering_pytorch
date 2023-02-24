import torch


class Param(object):
    def __init__(self):
        super(Param, self).__init__()
        # Path
        self.ROOT = 'D:/[논문]/[3]'
        self.DATASET_PATH = f'{self.ROOT}/DB/2nd_Proposed/1-fold'
        self.OUTPUT_CKP = f'{self.ROOT}/IIC/backup/ckp'
        self.OUTPUT_LOG = f'{self.ROOT}/IIC/backup/log'
        self.CKP_LOAD = False
        self.LOAD_CKP_EPCOH = 0

        # Data
        self.INPUT_SIZE = 224
        self.AUG = False

        # Train or Test
        self.NumClasses = 2
        self.AuxClass = 256
        self.NumSubHeads = 1
        self.HeadList = ['A', 'B']
        self.EPOCH = 300
        self.HeadEpoch = 1

        self.tr_name_type_list = [
            ['MainTrain', 'Lamb_Loss'],
            ['MainTrain', 'No_Lamb_Loss'],
            ['AuxTrain', 'Lamb_Loss'],
            ['AuxTrain', 'No_Lamb_Loss'],
        ]

        self.valid_name_type_list = [
            ['Valid', 'Accuracy']
        ]

        self.Visualization = True


        self.LR = 0.01
        self.BATCHSZ = 16
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PRINT_DISPLAY = True

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0
