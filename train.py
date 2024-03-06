import torch
from tqdm import tqdm
from datetime import datetime
import numpy as np
import utils
from PIL import Image

def validate(model, device, loader, metrics):
    metrics.reset()
    model.eval()
    model.to(device)

    with torch.no_grad():
        for (inputs, labels) in tqdm(loader):
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            output = model(inputs)

            #predictions see deeplabv3 doc
            preds = output['out'][0].argmax(0).unsqueeze(0).cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

def train_one_epoch(model, device, optimizer, loss_fn,scheduler, train_loader):
    running_loss = 0.
    last_loss = 0.
    model.to(device)

    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        for i,data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(inputs)['out']

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            #statistics
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 
                print('batch {} loss: {}'.format(i+1,last_loss))
                running_loss = 0.
            pbar.update(1)
        scheduler.step()
    return last_loss

def training(model, device, epochs, optimizer, loss_fn, scheduler, train_loader, val_loader):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1000000
    loss = []
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch(model, device, optimizer, loss_fn, scheduler, train_loader)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for vinputs, vlabels in val_loader:
                vinputs = vinputs.to(device, dtype=torch.float32)
                vlabels = vlabels.to(device, dtype=torch.long)

                voutputs = model(vinputs)['out']
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
        
        avg_vloss = running_vloss / len(val_loader)
        loss.append(avg_loss)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'resnet_no_scaling_colorjitter_{}_{}'.format(timestamp, epoch+1)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': loss}, model_path)
            #torch.save(model.state_dict(), model_path)
    model_path = 'resnet_no_scaling_colorjitter_last'
    torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': loss}, model_path)

def visualize(model, device, loader):
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    model.eval()
    model.to(device)
    with torch.no_grad():
        images, labels = next(iter(loader))

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        #image = get_adaption_inputs(image, tr_transform_adapt, device)

        output = model(images)

    preds = output['out'].max(dim=1)[1].cpu().numpy()
    targets = labels.cpu().numpy()

    for i in range(len(images)):
        image = images[i].detach().cpu().numpy()
        target = targets[i]
        pred = preds[i]

        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
        decode_target = loader.dataset.decode_target(target).astype(np.uint8)
        decode_pred = loader.dataset.decode_target(pred).astype(np.uint8)

        Image.fromarray(image).save('results/test_image_{}.png'.format(i))
        Image.fromarray(decode_target).save('results/test_target_{}.png'.format(i))
        Image.fromarray(decode_pred).save('results/test_pred_{}.png'.format(i))

