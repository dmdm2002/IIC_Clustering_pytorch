import torch


class Param(object):
    def __init__(self):
        super(Param, self).__init__()
        # Path
        self.ROOT = 'C:/Users/rlawj'
        self.DATASET_PATH = f'{self.ROOT}/sample_DB/catdog'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/ckp'
        self.OUTPUT_LOG = f'{self.ROOT}/backup/log'
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


        self.LR = 0.01
        self.BATCHSZ = 1
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PRINT_DISPLAY = True

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0