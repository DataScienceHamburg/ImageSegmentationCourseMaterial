#%% packages
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sem_seg_dataset import SegmentationDataset
import segmentation_models_pytorch as smp 
import torchmetrics
# %% Dataset and Dataloader
