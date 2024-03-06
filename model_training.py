import torch
import torchvision
from torch.utils import data
from utils import ext_transforms as et
from datasets import Cityscapes
from utils import loss
from train import training,validate,visualize
import os
from metrics import StreamSegMetrics

# train_transform = et.ExtCompose([
#     et.ExtRandomScale(scale_range=(0.5, 2.0)),
#     et.ExtRandomCrop(size=(768,768)),
#     et.ExtRandomHorizontalFlip(),
#     et.ExtToTensor(),
#     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225])
# ])

#train_transform = et.ExtCompose([
#    et.ExtRandomScale(scale_range=(0.5, 2.0)),
#    et.ExtRandomCrop(size=(512,1024)),
#    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#    et.ExtRandomHorizontalFlip(),
#    et.ExtToTensor(),
#    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                    std=[0.229, 0.224, 0.225])
#])

train_transform = et.ExtCompose([
    et.ExtRandomCrop(size=(512,1024)),
    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    et.ExtRandomHorizontalFlip(),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
])

val_transform = et.ExtCompose([
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
])

train_set = Cityscapes(root='/vol/tmp/phanfran/cityscapes/',
                    subset= None,
                    split='train',
                    transform=train_transform)

val_set = Cityscapes(root='/vol/tmp/phanfran/cityscapes/',
                       subset=None,
                       split='val',
                       transform=val_transform)

def main():
    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=19)
    device = 'cuda:1' if torch.cuda.is_available else 'cpu'
    model = torch.nn.DataParallel(model, device_ids=[1,0])
    model.to(device)

    BATCH_SIZE_TRAIN = 8
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True, num_workers=8)
    BATCH_SIZE_VAL = 1
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=True)
    #loss_fn = loss.FocalLoss(ignore_index=255, size_average=True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255,reduction='mean')

    optimizer = torch.optim.SGD(params=model.module.parameters(), lr=0.07, weight_decay=3e-3)
    #optimizer = torch.optim.SGD(params=model.module.parameters(), lr=0.07, weight_decay=1e-4)
    #optimizer = torch.optim.SGD(params=[
    #    {'params': model.module.backbone.parameters(), 'lr': 0.1 * 0.01},
    #    {'params': model.module.classifier.parameters(), 'lr': 0.01},
    #],lr=0.01, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(verbose=True, optimizer=optimizer, power=0.9,total_iters=200)

    EPOCHS=200

    torch.cuda.empty_cache()

    training(model=model,device=device,epochs=EPOCHS,optimizer=optimizer,loss_fn=loss_fn,scheduler=scheduler,train_loader=train_loader,val_loader=val_loader)


if __name__ == '__main__':
    main()

