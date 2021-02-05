import models
#import torch
import datetime

if __name__ == '__main__':

#    num_classes=10
#    model, input_size = models.pretrained.initialize_model('vgg-16', feature_extract=True, num_classes=num_classes)
#    print(model)
#
#    linear = models.classifiers.ClassifierVGG(model, num_classes, num_tries=10 , keep_ratio=0.5 )
#
#    x = torch.randn((2, 3, 224, 224))
#
#    out = linear(x)

    date = datetime.date.today().strftime('%Y%m%d')
    print(date)
