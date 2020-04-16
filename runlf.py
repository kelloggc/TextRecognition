import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lf
from lf import lf_dataset, lf_loss
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import numpy as np
import cv2
import sys
import json
import os
import yaml

from utils.continuous_state import init_model
from utils.dataset_parse import load_file_list
from utils import safe_load

with open(sys.argv[1]) as f:
    config = yaml.load(f)

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']
imageswritten = 0
test_set_list = load_file_list(pretrain_config['validation_set'])
dtype = torch.FloatTensor
line_follower = LineFollower()
state = safe_load.torch_state("/Users/carolinekellogg/Desktop/start_follow_read-master/data/snapshots/best_validation/lf.pt")
line_follower.load_state_dict(state)
test_dataset = LfDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False, num_workers=0,
                             collate_fn=lf_dataset.collate)
for step_i, x in enumerate(test_dataloader):
    x = x[0]
    positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyrs']]
    xy_positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyxy']]
    img = Variable(x['img'].type(dtype), requires_grad=False)[None,...]

    if len(xy_positions) <= 1:
        print "Skipping"
        continue

    grid_line, _, _, xy_output = line_follower(img, positions[1:], steps=len(positions), skip_grid=False)

    line = torch.nn.functional.grid_sample(img.transpose(2,3), grid_line)
    line = (line + 1.0) * 128
    cv2.imwrite("/Users/carolinekellogg/Desktop/compsci/lf/%d.jpg" % imageswritten, line.data[0].cpu().numpy().transpose())
    imageswritten += 1
