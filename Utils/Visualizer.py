# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
#
# from Utils.Option import Param
#
#
# class Visualizer(Param):
#     def __init__(self):
#         super(Visualizer, self).__init__()
#
#     def visulization(self, value):
#         h = .02
#         x_min, x_max = 0, 1
#         y_min, y_max = 0, 1
#
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#         positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.revel())))).to(self.DEVICE)
#
#         fig, ax = plt.subplot()
#         value = value.detach()
#         value = value.cpu().numpy()
#         value = value.reshape(xx.shape)
#
#         ax.imshow(value, vmin=0, vmax=1)
#         ax.axis('off')
#         ax.scatter(50*X[:,0], 50*X[:1])