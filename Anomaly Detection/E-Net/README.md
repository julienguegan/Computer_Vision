
# PyTorch-ENet

PyTorch (v1.0.0) implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147), the major part of the repository is from [here](https://github.com/davidtvs/PyTorch-ENet/tree/master/save).
The majors changements are the ``main.ipynb`` file that run on google colab, ``temperature_scaling.py`` and some minors updates in the pre-existent files. Some models have been saved in (https://github.com/julienguegan/Computer_Vision/tree/master/Anomaly%20Detection/E-Net)/save) but without any real results.


## Project structure

### Folders

- [``data``](https://github.com/davidtvs/PyTorch-ENet/tree/master/data): Contains instructions on how to download the datasets and the code that handles data loading.
- [``metric``](https://github.com/davidtvs/PyTorch-ENet/tree/master/metric): Evaluation-related metrics.
- [``models``](https://github.com/davidtvs/PyTorch-ENet/tree/master/models): ENet model definition.
- [``save``](https://github.com/davidtvs/PyTorch-ENet/tree/master/save): By default, ``main.py`` will save models in this folder. The pre-trained models can also be found here.

### Files
- [``main.ipynb``](https://github.com/julienguegan/Computer_Vision/blob/master/Anomaly%20Detection/E-Net/main.ipynb): Notebook running on Google colab 
- [``temperature_scaling.py``](https://github.com/julienguegan/Computer_Vision/blob/master/Anomaly%20Detection/E-Net/temperature_scaling.py): Class that allow temperature scaling on a neural network module (PyTorch)
- [``args.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/args.py): Contains all command-line options.
- [``main.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/main.py): Main script file used for training and/or testing the model.
- [``test.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/test.py): Defines the ``Test`` class which is responsible for testing the model.
- [``train.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/train.py): Defines the ``Train`` class which is responsible for training the model.
- [``transforms.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/transforms.py): Defines image transformations to convert an RGB image encoding classes to a ``torch.LongTensor`` and vice versa.
