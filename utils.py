# NOTE: to add normalization and denormalization

import torchvision.transforms as tf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import os

# Dataset class
class Detection_Dataset(Dataset):
  def __init__(self, annots_json, gt_reduce=None, transforms=None):
    assert os.path.exists(annots_json), 'Error: Annotation json file does not exist!'
    assert isinstance(transforms, list), 'Error: transforms must be in a list!'
    self.annots = self.read_json(annots_json)
    self.num_data = len(self.annots['data'])
    self.num_classes = self.annots['num_classes']
    self.gt_reduce = gt_reduce if gt_reduce!=None else self.annots['gt_reduce']
    self.tfm = tf.Compose([tf.ToTensor()] if transforms==None else transforms)

  def read_json(self, annots_json):
    # annots keys:
    # 'images_root', 'annots_root', 'labelmap_file', 'num_classes', 'gt_reduce', 'data'
    with open(annots_json, 'r') as jsonfile:
      annots_dict = json.loads(jsonfile.read())
    self.im_root = annots_dict['images_root']
    assert os.path.exists(self.im_root), 'Error: Image root folder does not exist!'
    return annots_dict

  def gen_maps(self):
    # annot row format:
    # [im_name, im_w, im_h, num_objs, [idx, ct_x, ct_y, off_x, off_y, bbox_w, bbox_h], ...]
    # ch0: class 0, ch1: class 1, ...
    heatMap = torch.zeros(self.num_classes,
                          int(self.annot_row[2]/self.gt_reduce),
                          int(self.annot_row[1]/self.gt_reduce))
    # ch0: off_x, ch1: off_y
    offsMap = torch.zeros(2,
                          int(self.annot_row[2]/self.gt_reduce),
                          int(self.annot_row[1]/self.gt_reduce))
    # ch0: bbox_w, ch1: bbox_h
    dimsMap = torch.zeros(2,
                          int(self.annot_row[2]/self.gt_reduce),

                          int(self.annot_row[1]/self.gt_reduce))
    for j in range(self.annot_row[3]): # iterate for number of objs
      id, ct_x, ct_y, of_x, of_y, bb_w, bb_h = self.annot_row[4+j]
      heatMap[id, ct_y, ct_x] = 1
      offsMap[ :, ct_y, ct_x] = torch.FloatTensor((of_x, of_y))
      dimsMap[ :, ct_y, ct_x] = torch.FloatTensor((bb_w, bb_h))
    indsMap = heatMap.clone()
    return heatMap, offsMap, dimsMap, indsMap

  def get_sample_row(self):
    return self.annots['data'][np.random.randint(self.num_data)]

  def __getitem__(self, i):
    # Returns the obj det maps for the 3 heads
    self.annot_row = self.annots["data"][i]
    #img_tsr = self.tfm(Image.open(f'{self.im_root}/{self.annots["data"][i][0]}'))
    img_tsr = self.tfm(Image.open(f'{self.im_root}/{self.annot_row[0]}'))
    heatMap, offsMap, dimsMap, indsMap = self.gen_maps()
    return {'image'  : img_tsr,
            'heatMap': heatMap,
            'offsMap': offsMap,
            'dimsMap': dimsMap,
            'inds'   : indsMap}

  def __len__(self):
    return self.num_data

# Extract labels from labelmap file
def get_labels(labelmap):
  assert os.path.exists(labelmap), 'Error: Labelmap file does not exist!'
  with open(labelmap, 'r') as lbfile:
    labels = lbfile.readlines()
  #labels = [lb.strip('\n').split(' ')]
  return {int(lb.strip('\n').split(' ')[0]):lb.strip('\n').split(' ')[1] for lb in labels}

# Visualize annots/preds
def visualize(dataDict, labelDict, gt_reduce=4, conf_thresh=0.5, cv2Disp=True):
  assert isinstance(dataDict, dict), 'Error: input must be a dictionary of image and det maps!'
  image = (dataDict['image'].detach().numpy()[0]*255).clip(0,255).transpose(1,2,0).astype(np.uint8)
  ctrs = np.where(dataDict['heatMap'].detach().numpy()[0]>=conf_thresh)
  print('centers:',ctrs)
  offs = dataDict['offsMap'].detach().numpy()[0]
  dims = dataDict['dimsMap'].detach().numpy()[0]
  for i in range(len(ctrs[0])): # iterate over all objects
    idx = ctrs[0][i]
    ctr_y = ctrs[1][i]
    ctr_x = ctrs[2][i]
    conf = dataDict['heatMap'].detach().numpy()[0,idx,ctr_y,ctr_x]
    off_x = offs[0, ctr_y, ctr_x]
    off_y = offs[1, ctr_y, ctr_x]
    bbox_w = dims[0, ctr_y, ctr_x]
    bbox_h = dims[1, ctr_y, ctr_x]
    #print(f'id:{idx}\nctr_y:{ctr_y} ctr_x:{ctr_x}\noff_y:{off_y} off_x:{off_x}\nbbox_w:{bbox_w} bbox_h:{bbox_h}')
    x_ctr = gt_reduce*(ctr_x+off_x)
    y_ctr = gt_reduce*(ctr_y+off_y)
    x_min = int(x_ctr-bbox_w/2)
    x_max = int(x_ctr+bbox_w/2)
    y_min = int(y_ctr-bbox_h/2)
    y_max = int(y_ctr+bbox_h/2)
    #print(f'xmin:{x_min}, ymin:{y_min}, xmax:{x_max}, ymax:{y_max}.')
    image = cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0,255,0), 1)
    image = cv2.putText(image, f'{labelDict[idx]} {conf:.2f}', (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1, cv2.LINE_AA)
  if cv2Disp:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

# Losses
def variant_focal_loss(pred_batch, target, alpha=2, beta=4):
  # Modified focal loss for Objects as Points paper
  pos_inds = target.eq(1).float()
  neg_inds = target.lt(1).float()
  neg_weights = torch.pow(1 - target, beta)
  loss = 0
  # iterate across batch size
  for pred in pred_batch:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1-1e-4)
    pos_loss = torch.log(pred) * torch.pow(1-pred, alpha) * pos_inds
    neg_loss = torch.log(1-pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    num_pos = pos_inds.float.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos==0: # no dets
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(pred_batch)

def l1_reg_loss(pred_batch, target, gt_inds):
  # only assess targeted areas from heatmap
  return F.l1_loss(pred_batch*gt*inds, target)

if __name__ == '__main__':
  labelDict = get_labels('data/moneky_labels.txt')
  trainset = Detection_Dataset('monkey_train.json')
  trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
  print('Dataset length:', len(trainloader.dataset))
  print('Sample row:\n  ',trainset.get_sample_row())
  for i, data in enumerate(trainloader):
    print(data['image'].shape)
    print(np.where(data['heatMap'].detach().numpy()[0]>0.5))
    print(data['offsMap'].shape)
    print(data['dimsMap'].shape)
    img_out = visualize(data, labelDict)
    #plt.figure()
    #plt.imshow(img_out.get())
    #plt.title('Detection Output')
    #plt.show()
    cv2.imshow('Detection Output', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
