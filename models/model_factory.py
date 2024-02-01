# This module creates models for our experiments.
import timm
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights
import model_functions
import torch.nn as nn

def get_model(pretrained_model_name, classifier_fn, layers_version, trainable_layers=None):

    if pretrained_model_name == "xception":
        model = timm.create_model(pretrained_model_name, pretrained=True)
        model.fc = classifier_fn(model.fc.in_features)
        
    elif pretrained_model_name == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = classifier_fn(model.fc.in_features)

    elif pretrained_model_name == "densenet121":
        model = densenet121(pretrained=True)
        model.classifier = classifier_fn(model.classifier.in_features)

    else:
        raise ValueError(f"Unsupported model: {pretrained_model_name}")

    if trainable_layers is None:
        trainable_layers = predefined_trainable_layers[pretrained_model_name][layers_version](model)

    model_functions.set_trainable_layers(model, trainable_layers)

    return model    

# Define a dictionary with functions returning predefined trainable layers for each model and version
predefined_trainable_layers = {
    "xception": {
        'classifier': lambda model: [model.fc],
        'first': lambda model: [model.fc, model.bn4, model.conv4],
        'second': lambda model: [model.fc, model.bn4, model.conv4, model.bn3, model.conv3],
    },
    "resnet50": {
        'classifier': lambda model: [model.fc],
        'first': lambda model: [model.fc, model.layer4[2].bn3, model.layer4[2].conv3, model.layer4[2].bn2, model.layer4[2].conv2],
        'second': lambda model: [model.fc, model.layer4[2].bn3, model.layer4[2].conv3, model.layer4[2].bn2, model.layer4[2].conv2, model.layer4[2].bn1, model.layer4[2].conv1, model.layer4[1].bn3, model.layer4[1].conv3],
    },
    "densenet121": {
        'classifier': lambda model: [model.classifier],
        'first': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10],
        'second': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10, model.features.denseblock4.denselayer9, model.features.denseblock4.denselayer8, model.features.denseblock4.denselayer7, model.features.denseblock4.denselayer6],
    }
}

def get_linear_classifer(n_inputs, num_classes=2):
    return nn.Sequential(
    nn.Linear(n_inputs, num_classes),
    nn.LogSoftmax(dim=1))

def get_simple_non_linear_classifier(n_inputs, num_classes=2):
    return nn.Sequential(
    nn.BatchNorm1d(n_inputs),
    nn.Linear(n_inputs, n_inputs // 2),
    nn.ReLU(),
    nn.BatchNorm1d(n_inputs // 2),
    nn.Dropout(0.5),
    nn.Linear(n_inputs // 2, num_classes),
    nn.LogSoftmax(dim=1) 
)



    




        

        


