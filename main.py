from RunModules.trainer import Train
from Utils.Option import Param


class handler(Param):
    def __init__(self):
        super(handler, self).__init__()

    def start(self):
        if self.run_type == 0:
            tr = Train()
            tr.run()


hand = handler()
hand.start()