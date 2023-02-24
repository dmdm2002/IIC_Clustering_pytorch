import os
from Utils.Functinos import TensorboardHandler
from Utils.ErrorLogger import error_logger


class LossAccDisplayer:
    def __init__(self, name_type_list):
        """
        :param name_type_list: 우리가 확인하기를 원하는 losses 의 이름 list과 run_type ex) ['train', 'acc']

        :var self.count: 총 iter 의 횟수
        :var self.value_list: loss values 가 저장될 list
        """
        self.place = os.path.realpath(__file__)
        self.count = 0
        self.name_type_list = name_type_list
        self.value_list = [0] * len(self.name_type_list)
        self.tensorboard_handler = TensorboardHandler()

    def record(self, values, select_index=False):
        self.count += 1
        try:
            if select_index:
                for index, value in values:
                    self.value_list[index] += value
            else:
                for i, value in enumerate(values):
                    self.value_list[i] += values[i]
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e=e)

    def get_avg_losses(self, length):
        try:
            return [value / length for value in self.value_list]
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e=e)

    def logging_info(self, ep):
        try:
            self.tensorboard_handler.global_logging(self.value_list, self.name_type_list, ep)
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e=e)

    def reset(self):
        self.count = 0
        self.value_list = [0] * len(self.name_type_list)