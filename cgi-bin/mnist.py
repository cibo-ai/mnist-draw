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

# Default output
res = {"result": 0,
       "data": [], 
       "error": '',
       "output": ''}

try:
  # Get post data
  if os.environ["REQUEST_METHOD"] == "POST":
    data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

    # Convert data url to numpy array
    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes)
    arr = np.array(im)[:,:,0:1]

    # Normalize and invert pixel values to range [-1, 1]
    arr = -1. + (255. - arr)*2 / 255.

    # transform numpy array into a tensor
    arr = torch.from_numpy(arr)
    arr = torch.reshape(arr, (1, 28, 28))
    arr = arr.unsqueeze(0)

    # Load the model
    model = torch.load('/home/crow/handwritten-digit/models/conv2d_model.pt')
    model.eval()
    # Change the model to double type because arr is double for some reason
    model.double()

    # get the prediction from the model
    with torch.no_grad():
      logps = model(arr)

    # get the prediction result in list of probable from 0-9
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])

    res['result'] = 1
    res['data'] = probab
    
    # output to the browser dev console for debugging
    res['output'] = 'debug text'

except Exception as e:
  # Return error data
  res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))

