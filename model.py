import torch
from torch import nn, optim
from torchvision import models

def build_model(architecture, hidden_units, class_to_idx):
    if architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif architecture == 'vgg11':
        model = models.vgg11(pretrained=True)
        input_size = 25088
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif architecture == 'densenet169':
        model = models.densenet169(pretrained=True)
        input_size = 1664
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = 2208
    elif architecture == 'densenet201':
        model = models.densenet201(pretrained=True)
        input_size = 1920
    else:
        raise ValueError("architecture not supported")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build custom classifier
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, len(class_to_idx)),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    return model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['architecture'], 512, checkpoint['class_to_idx'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['class_to_idx']

def predict(image, model, topk, gpu):
    model.eval()
    image = image.unsqueeze(0).float()
    if gpu:
        image = image.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)

    probs, indices = ps.topk(topk)
    probs, indices = probs.cpu().numpy()[0], indices.cpu().numpy()[0]

    return probs, indices
