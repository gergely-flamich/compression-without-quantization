# Compression without Quantization

Compression without quantization is a lossy transform coding framework for compression an arbitrary real-valued tensor x.

This repo presents an application of this framework to lossy image compression. A fully convolutional Probabilistic Ladder Network is implemented in Tensorflow (TF), Tensorflow Probability (TFP) and Tensorflow Compression (TFC) that can be trained and used on an arbitrary image dataset.

The code is writtent in Python 3.6.

## Setup

Clone this repository
```
> git clone https://github.com/gergely-flamich/compression-without-quantization.git
> cd compression-without-quantization
```

Create a virtual environment and activate it
```
> virtualenv cwoq-venv
> source cwoq-venv/bin/activate
```

Install the requirements (note that this may take a while)
```
(cwoq-venv)> pip install -r requirements.txt
```

To train / compress / decompress and to see what options are available for these, simply run
```
(cwoq-venv)> python code/miracle.py --help
```

## Queries

Please contact the author at flamich.gergely@gmail.com