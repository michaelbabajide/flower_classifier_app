import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

def load_data(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 5, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 5, shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 5, shuffle=True)}

    class_to_idx = image_datasets['train'].class_to_idx
    return dataloaders, class_to_idx

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    
    image.thumbnail((224,224))
    width, height = image.size
    left = (width - 224)/2
    right = (width + 224)/2
    top = (height - 224)/2
    bottom = (height + 224)/2
    image.crop((left, top, right, bottom))

    image = np.array(image)/255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.485, 0.456, 0.406])
    image = (image - mean)/std
    image.transpose((2,0,1))

    return torch.from_numpy(image)
