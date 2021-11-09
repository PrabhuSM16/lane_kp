import cv2
import matplotlib.pyplot as plt
import numpy as np

def gauss_radius(det_size, min_overlap=0.7):
  # det_size contains bbox height and width
  h, w = det_size
  a1 = 1
  b1 = h+w
  c1 = w*h*(1-min_overlap)/(1+min_overlap)
  sq1 = np.sqrt(b1**2 - 4*a1*c1)
  r1 = (b1-sq1) / (2*a1)

  a2 = 4
  b2 = 2*(h+w)
  c2 = w*h*(1-min_overlap)
  sq2 = np.sqrt(b2**2 - 4*a2*c2)
  r2 = (b2-sq2) / (2*a2)

  a3 = 4*min_overlap
  b3 = -2*min_overlap*(h+w)
  c3 = w*h*(min_overlap-1)
  sq3 = np.sqrt(b3**2 - 4*a3*c3)
  r3 = (b3-sq3)/2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
  # m, n = [(ss-1.)/2. for ss in shape]
  m, n = shape # insert (radius_x, radius_y) for shape
  y, x = np.ogrid[-m:m+1, -n:n+1] # generate Gaussian coordinates
  h = np.exp(-(x**2 + y**2)/ (2*sigma**2)) # compute height of Gaussian
  h[h < np.finfo(h.dtype).eps * h.max()] = 0 # ensures all values >=0
  return h

def draw_hmap(ctr_map, map_size=(120,160), kp_thresh=0, gauss_iou=0.7, k=1):
  h, w = map_size
  hmap = np.zeros((h, w), dtype=np.float32)
  radius = max(0, int(gauss_radius((math.ceil(h), math.ceil(w)), gauss_iou)))
  #diameter = 2*radius+1
  gaussian = gaussian2D((radius, radius), sigma=diameter/6)
  y, x = np.where(ctr_map>kp_thresh)
  if ctr_map.shape != map_size:
    x. y = [(int(i*map_size[0]/w), int(j*map_size[1]/h)) for i, j in zip(x, y)]
  left, right = min(x, radius), min(width-x, radius+1)
  top, bottom = min(y, radius), min(height-y, radius+1)
  masked_hmap = hmap[y-top:y+bottom, x-left:x+right]
  masked_gauss = gaussian[radius-top:radius+bottom, radius-left:radius+right]
  if min(masked_gauss.shape)>0 and min(masked_hmap.shape)>0:
    np.maximum(masked_hmap, masked_gauss*k, out=masked_hmap)
  return hmap
