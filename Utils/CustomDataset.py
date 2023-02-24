import os
import pandas as pd

from torch.utils import data
from PIL import Image

from Utils.ErrorLogger import error_logger


class Datasets(data.Dataset):
    def __init__(self, db_path, run_type, transform=None, aug_transform=None):
        """
        DataLoader Custom DataSet

        :param db_path: DATABASE PATH
        :param run_type: Model run type (train, test, valid)
        :param transform: Image style transform module
        :var _data_info: Dataframe with DB information for us to use [folder, image name, class]
        :var self.path_list: Data Path information list
        :var self.label_list: Data class information list
        """
        super(Datasets, self).__init__()
        self.place = os.path.realpath(__file__)
        self.db_path = db_path

        self.run_type = run_type
        assert self.run_type == 'train' or self.run_type == 'test' or self.run_type == 'valid', \
            'Only train, test, and valid are available for run_type.'

        self.transform = transform
        self.aug_transform = transform

        _data_info = None
        try:
            _data_info = pd.read_csv(f'{self.db_path}/{self.run_type}.csv')
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e)

        self.path_list = self.get_path(_data_info)[:5]
        if run_type == 'test' or run_type == 'valid' or run_type == 'train':
            self.label_list = self.get_labes(_data_info)[:5]

    def get_path(self, data_info):
        path_list = []
        path_info = data_info.iloc[:, :2].values
        for folder, name in path_info:
            full_path = f'{self.db_path}/{self.run_type}/{folder}/{name}'
            path_list.append(full_path)

        return path_list

    def get_labes(self, data_info):
        return data_info.iloc[:, 2:].values

    def get_ImageName(self, path):
        ImageName = path.split('/')[-1]
        ImageName = ImageName.split('.')[0]

        return ImageName

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if self.aug_transform is None:
            item = self.transform(
                Image.open(self.path_list[idx])
            )
            label = self.label_list[idx]
            image_name = self.get_ImageName(self.path_list[idx])

            return [item, label, image_name]

        else:
            item = self.transform(
                Image.open(self.path_list[idx])
            )
            item_aug = self.aug_transform(
                Image.open(self.path_list[idx])
            )
            label = self.label_list[idx]
            image_name = self.get_ImageName(self.path_list[idx])

            return [item, item_aug, label, image_name]