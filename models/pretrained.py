from torchvision import models, datasets, transforms
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):

	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

def initialize_model(model_name, feature_extract, use_pretrained=True):

    if model_name == 'vgg':
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 1)
        input_size = 224

    return model_ft, input_size

