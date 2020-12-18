# MNIST-draw
This repository contains a single page website that enables user to hand-draw and classify digits (0-9) using machine leanring. A machine learning model trained against the MNIST dataset is used for classification.

This repository is a forked from rhammell/mnist-draw that instead use PyTorch library to build and train the model.

## Setup
Python 3.6+ is recommended for compatability with all required modules.

Require Numpy, PyTorch, Pillow and SciPy.

## Usage
To launch the website, begin by starting a Python server from the repository folder:
```bash
# Start Python server
python -m http.server --cgi 8000
```

Then open the browser and navigate to `http://localhost:8000/index.html` to view it.

Users are guided to draw a digit (0-9) on an empty canvas then hit the 'Predict" button to process their drawing. Allow up to 30 second for the process to complete. Any error during processing will be indicated with a warning icon and printed to the console.

Results are displayed as a bar graph where each classification label receives a score between 0.0 to 1.0 from the machine learning model. CLear the canvas with the 'Clear' button to draw and process other digits.

## Machine Learning Model
A Python script related to drawing and feeding the data to the machine learning model is contained within the `cgi-bin` folder.

A Jupyter notebook `build_model.ipynb` contains a script to download, load and save the MNIST dataset to be use as input, a convolutional neural network (CNN) using PyTorch nn module. Trained model's parameter files are saved into the `models` directory. Pre-trained model files are available in this directory already, as well as a simple 3 dense layers linear model.

The `mnist.py` script implements this trained model against the user's hand-draw input. When the 'Predict' button is clicked, the contents of the drawing canvs are posted to this script as data url, and a JSON object containing the model's predictions are returned.
