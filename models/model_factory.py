# This module creates models for our experiments.
import timm
import model_functions
import torch.nn as nn

# default training configuration
config = {
    'batch_size': 128,
    'val_split': 0.1,
    'lr': 1e-3,
    'n_epochs': 20,
    'image_size': None,
    'mean': None,
    'std': None
    }

def get_model(pretrained_model_name, classifier_fn, layers_version, trainable_layers=None, n_epochs=10):

    def set_classifier():
        if pretrained_model_name in ["xception", "resnet50", "inception_v3"]:
            model.fc = classifier_fn(model.fc.in_features)
        else:
            model.classifier = classifier_fn(model.classifier.in_features)

    model = timm.create_model(pretrained_model_name, pretrained=True)
    set_classifier()

    config['mean'] = model.default_cfg['mean']
    config['std'] = model.default_cfg['std']
    config['image_size'] = model.default_cfg['input_size'][1]
    config['n_epochs'] = n_epochs

    if trainable_layers is None:
        trainable_layers = predefined_trainable_layers[pretrained_model_name][layers_version](model)

    model_functions.set_trainable_layers(model, trainable_layers)

    return model, config

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
    },

    "inception_v3": {
        'classifier': lambda model: [model.fc],
        # first and second are not working yet
        'first': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10],
        'second': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10, model.features.denseblock4.denselayer9, model.features.denseblock4.denselayer8, model.features.denseblock4.denselayer7, model.features.denseblock4.denselayer6],
    },
    "efficientnet_b1": {
        'classifier': lambda model: [model.classifier],
        # first and second are not working yet
        'first': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10],
        'second': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10, model.features.denseblock4.denselayer9, model.features.denseblock4.denselayer8, model.features.denseblock4.denselayer7, model.features.denseblock4.denselayer6],
    },
}



    




        

        


