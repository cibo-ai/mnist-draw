#!/usr/bin/env python
"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""
import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from scipy import ndimage
import math

# Default output
res = {"result": 0,
       "data": [], 
       "error": '',
       "output": ''}

# Load the model
model = torch.load('/home/crow/handwritten-digit/models/conv2d_model.pt')
model.eval()
# Change the model to double type because image is double for some reason
model.double()

def getBestShift(image):
  # Get the shift value for the transformation matrix
  cy, cx = ndimage.measurements.center_of_mass(image)
  rows, cols = image.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)
  return shiftx, shifty

def shift(image, sx, sy):
  # Shift the image
  rows, cols = image.shape
  M = np.float32([[1, 0, sx], [0, 1, sy]])
  return ndimage.affine_transform(image, matrix=M, output_shape=(cols, rows))
 
def preprocessImage(image):
  # Remove every rows and cols that are empty (-1)
  while np.sum(image[0]) == -28:
    image = image[1:]
  while np.sum(image[:,0]) == -28:
    image = np.delete(image, 0, 1)

  while np.sum(image[-1]) == -28:
    image = image[:-1]
  while np.sum(image[:,-1]) == -28:
    image = np.delete(image, -1, 1)
  
  rows, cols = image.shape

  # Fill the missing -1 rows and column with -1
  colsPadding = (int(math.ceil((28-cols)/2.0)), int(math.floor((28-cols)/2.0)))
  rowsPadding = (int(math.ceil((28-rows)/2.0)), int(math.floor((28-rows)/2.0)))
  image = np.lib.pad(image, (rowsPadding, colsPadding), mode='constant', constant_values = -1)
  #shiftx, shifty = getBestShift(image)
  #image = shift(image, shiftx, shifty)
  return image

try:
  # Get post data
  if os.environ["REQUEST_METHOD"] == "POST":
    data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

    # Convert data url to numpy array
    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes)
    image = np.array(im)[:,:,0]
    image = np.reshape(image, (28, 28))

    # Normalize and invert pixel values to range [-1, 1]
    image = -1 + (255. - image) * 2 / 255.

    # Preprocess the image to look like MNIST training dataset
    image = preprocessImage(image)

    # transform numpy array into a tensor
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)

    # get the prediction from the model
    with torch.no_grad():
      logps = model(image)

    # get the prediction result in list of probable from 0-9
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])

    res['result'] = 1
    res['data'] = probab
    
    # output to the browser dev console for debugging
    res['output'] = str(image) 

except Exception as e:
  # Return error data
  res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))

