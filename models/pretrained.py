from torchvision import models, datasets, transforms
import torch.nn as nn

def set_parameter_requires_grad(model, grad):

    for param in model.parameters():
        param.requires_grad = False

def initialize_model(model_name, pretrained, feature_extract=True, freeze=False, num_classes=10, use_pretrained=True) -> (nn.Module, int):

    if model_name.find('vgg') != -1:
        model_ft = models.vgg16(pretrained=pretrained)
        #model_ft.requires_grad(not feature_extract)
        #set_parameter_requires_grad(model_ft, not feature_extract)

        num_ftrs = model_ft.classifier[6].in_features  # change the last layer

        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        new_class_lst = [n for n in model_ft.classifier if not type(n) is nn.Dropout]  # remove dropout
        input_size = 224
        model_ft.classifier = nn.Sequential(*new_class_lst)

        set_parameter_requires_grad(model_ft, not freeze)

        if feature_extract:
            set_parameter_requires_grad(model_ft.features, False)

    elif model_name.find('resnet') != -1:

        pass

    return model_ft, input_size

