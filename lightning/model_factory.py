# This module creates models for our experiments.
import timm
import torch.nn as nn

def get_model(pretrained_model_name, classifier_fn, layers_version, trainable_layers=None):

    def set_classifier():
        if pretrained_model_name in ["xception", "resnet50"]:
            model.fc = classifier_fn(model.fc.in_features)
        else:
            model.classifier = classifier_fn(model.classifier.in_features)

    model = timm.create_model(pretrained_model_name, pretrained=True)
    set_classifier()

    img_config = dict()
    img_config['mean'] = model.default_cfg['mean']
    img_config['std'] = model.default_cfg['std']
    img_config['image_size'] = model.default_cfg['input_size'][1]

    if trainable_layers is None:
        trainable_layers = predefined_trainable_layers[pretrained_model_name][layers_version](model)

    set_trainable_layers(model, trainable_layers)

    return model, img_config

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

# Accepts a list of trainable layers. Makes sure that only these layers are not frozen.
def set_trainable_layers(model, trainable_layers):
    
    for param in model.parameters():
        param.requires_grad = False

    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} trainable parameters.')


# Define a dictionary with functions returning predefined trainable layers for each model and version
predefined_trainable_layers = {
    "xception": {
        'classifier': lambda model: [model.fc],
        'first': lambda model: [model.fc, model.bn4, model.conv4],
        'second': lambda model: [model.fc, model.bn4, model.conv4, model.bn3, model.conv3],
        'all': lambda model: [model]
    },

    "resnet50": {
        'classifier': lambda model: [model.fc],
        'first': lambda model: [model.fc, model.layer4[2].bn3, model.layer4[2].conv3, model.layer4[2].bn2, model.layer4[2].conv2],
        'second': lambda model: [model.fc, model.layer4[2].bn3, model.layer4[2].conv3, model.layer4[2].bn2, model.layer4[2].conv2, model.layer4[2].bn1, model.layer4[2].conv1, model.layer4[1].bn3, model.layer4[1].conv3],
        'all': lambda model: [model]
    },
    "densenet121": {
        'classifier': lambda model: [model.classifier],
        'first': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10],
        'second': lambda model: [model.classifier, model.features.norm5, model.features.denseblock4.denselayer16, model.features.denseblock4.denselayer15, model.features.denseblock4.denselayer14, model.features.denseblock4.denselayer13, model.features.denseblock4.denselayer12, model.features.denseblock4.denselayer11, model.features.denseblock4.denselayer10, model.features.denseblock4.denselayer9, model.features.denseblock4.denselayer8, model.features.denseblock4.denselayer7, model.features.denseblock4.denselayer6],
        'all': lambda model: [model]
    },

    "efficientnet_b0": {
        'classifier': lambda model: [model.classifier],
        'first': lambda model: [model.classifier, model.bn2, model.conv_head, model.blocks[6][0].bn3, model.blocks[6][0].conv_pwl],
        'second': lambda model: [model.classifier, model.bn2, model.conv_head, model.blocks[6]],
        'all': lambda model: [model]
    },
}



    




        

        


