import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn

from utils import safe_load

import sol
from sol.start_of_line_finder import StartOfLineFinder
from sol.alignment_loss import alignment_loss
from sol.sol_dataset import SolDataset
from sol.crop_transform import CropTransform

import lf
from lf import lf_dataset, lf_loss
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import numpy as np
import cv2
import json
import yaml
import sys
import os

from utils import transformation_utils, drawing
from utils.continuous_state import init_model
from utils.dataset_parse import load_file_list

with open(sys.argv[1]) as f:
    config = yaml.load(f)

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']

filenames = []
for filename in os.listdir('/Users/carolinekellogg/Desktop/compsci/deskew/thelittleprince/'):
    if (filename != ".DS_Store"):
        filenames.append("/Users/carolinekellogg/Desktop/compsci/deskew/thelittleprince/%s" % filename)
filenames.sort()
imageswritten=0
base0 = 16
base1 = 16
startofline = StartOfLineFinder(base0, base1)
state = safe_load.torch_state("/Users/carolinekellogg/Desktop/start_follow_read-master/data/snapshots/best_validation/sol.pt")
startofline.load_state_dict(state)
line_follower = LineFollower()
state = safe_load.torch_state("/Users/carolinekellogg/Desktop/start_follow_read-master/data/snapshots/best_validation/lf.pt")
line_follower.load_state_dict(state)
dtype = torch.FloatTensor

train_dataset = SolDataset(filenames,
                           rescale_range=pretrain_config['sol']['validation_rescale_range'],
                           transform=None)

train_dataloader = DataLoader(train_dataset,
                              batch_size=pretrain_config['sol']['batch_size'],
                              shuffle=True, num_workers=0,
                              collate_fn=sol.sol_dataset.collate)

starts = []

for step_i, x in enumerate(train_dataloader):
    img = Variable(x["img"].type(dtype), requires_grad=False)
    predictions = startofline(img)
    predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
    starts.append(predictions)

    """org_img = img[0].data.cpu().numpy().transpose([2,1,0])
    org_img = ((org_img + 1)*128).astype(np.uint8)
    org_img = org_img.copy()
    org_img = drawing.draw_sol_torch(predictions, org_img)
    cv2.imwrite("/Users/carolinekellogg/Desktop/compsci/sol_prince/%d.jpg" % imageswritten, org_img)
    imageswritten += 1"""

lf_files = []

for i in range(0, (len(filenames)-1)):
    temp = []
    temp.append(starts[i])
    temp.append(filenames[i])
    lf_files.append(temp)
