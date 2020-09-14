from torchvision import models, datasets, transforms
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):

	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

def initialize_model(model_name, feature_extract, num_classes=10, use_pretrained=True) -> (nn.Module, int):

    if model_name.find('vgg') != -1:
        model_ft = models.vgg16(pretrained=True)
        set_parameter_requires_grad(model_ft, not feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size

