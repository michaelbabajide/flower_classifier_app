# Flower Classifier Command Line Application

This command-line application is designed to train a neural network on a dataset and use the trained model to predict the class of a flower in an image. It consists of two main scripts:

- `train.py` for training a model and saving it as a checkpoint, and
- `predict.py` for predicting the class of an image using a trained model checkpoint.

## Content

- [Requirements](#requirements)
- [Usage](#usage)
    - [Training](#training)
    - [Prediction](#prediction)
- [Options](#options)
- [Files](#files)
- [License](#license)

## Requirements

The application requires Python 3.9 or higher and the following Python libraries:

- PyTorch
- torchvision
- Pillow (PIL)

## Usage

### Training

To train the a new neural networ, use 'train.py' with the following commands:

``` bash
python train.py data_directory [--save_dir 'checkpoint.pth'] [--arch 'vgg16'] [--learning_rate 0.001] [--hidden_units 512] [--epochs 20] [--gpu]
```

- 'data_directory': Path to the dataset directory containing the 'train', 'valid' and 'test' subdirectories.
- 'save_dir' (Optional): Directory where the trained model checkpoint will be saved. Default is 'checkpoint.pth'.
- 'arch' (Optional): Model architecture to use for training. Default is 'vgg16'.
- 'learning_rate' (Optional): Learning rate to use for training. Default is 0.001.
- 'hidden_units' (Optional): Number of hidden units in the classifier. Default is 512.
- 'epochs' (Optional): Number of training epochs. Default is 20.
- 'gpu' (Optional): Use GPU for training. Default is False.

### Prediction

To predict the class of a flower image, use 'predict.py' with the following command:

``` bash
python predict.py image_path checkpoint_path [--top_k 5] [--category_names 'cat_to_name.json'] [--gpu]
```

- 'image_path': Path to the image file for prediction.
- 'checkpoint_path': Path to the saved model checkpoint file.
- 'top_k' (Optional): Number of top most likely classes to return. Default is 5.
- 'category_names' (Optional): Path to a JSON file mapping the class values to real flower names. Default is 'cat_to_name.json'.
- 'gpu' (Optional): Use GPU for inference. Default is False.

## Options

You can customise your training by doing any of the following:

- You can choose different model architectures by specifying '--arch'. Supported model architectures include 'vgg', 'alexnet', and 'densenet'
- Adjust the learning rate with '--learning_rate' to control the training process.
- Adjust the number of hidden units in the classifier with '--hidden_units'
- Set the number of training epochs with '--epochs'
- Utilise the '--gpu' flag to enable GPU acceleration for both training and prediction.

## Files

- **train.py**: script for training a new neural network model
- **predict.py**: script for predicting the class of a flower image using a saved model checkpoint
- **model.py**: script containing functions for building, loading, and predicting with the neural network model
- **utils.py**: script containing utility functions for data loading and image preprocessing
- **cat_to_name.json**: JSON file mapping category indices to real flower names

## License

