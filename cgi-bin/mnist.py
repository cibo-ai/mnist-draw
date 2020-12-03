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

# Default output
res = {"result": 0,
       "data": [], 
       "error": '',
       "output": ''}

try:
  # Get post data
  if os.environ["REQUEST_METHOD"] == "POST":
    data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))
    print(json.dumps('hello'))

    # Convert data url to numpy array
    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes)
    arr = np.array(im)[:,:,0:1]
    
    # Normalize and invert pixel values
    arr = (255 - arr) / 255.
    arr[arr == 0.] = -1.

    # Load trained model
    model = torch.load('/home/crow/handwritten-digit/models/my_mnist_model.pt')
    model.double()

    # transform numpy array into a tensor
    arr = torch.from_numpy(arr).view(1, 784)
    a = arr.shape
    with torch.no_grad():
      logps = model(arr)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])

    res['result'] = 1
    res['data'] = probab
    res['output'] = str(arr.numpy())

except Exception as e:
  # Return error data
  res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


