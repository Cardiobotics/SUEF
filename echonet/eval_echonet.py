import torch
import numpy as np
import os
import torchvision
from run_echonet import run_epoch
from echonet_dataset import Echo
from dataset_utils import get_mean_and_std
from torch.cuda.amp import autocast
import tqdm
import sklearn.metrics
def main():
    """
    filename = 'test_result_echonet.csv'
    header = ['img_filepath', 'uid', 'instance_id', 'model_pred', 'target']
    with open(filename, 'w', encoding='UTF8') as f:
                            writer = csv.writer(f)
                            # write the header
                            writer.writerow(header)
    """
    model_name="r2plus1d_18"
    pretrained=False
    frames=32
    period=2
    num_workers=8
    batch_size=8
    device="cuda"
    path_to_model_weights = "echonet_pretrained_model/"

    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    if torch.cuda.is_available():
        print("cuda is available, original weights")
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(path_to_model_weights + "r2plus1d_18_32_2_pretrained.pt")
        model.load_state_dict(checkpoint['state_dict'])
    criterion = torch.nn.MSELoss()
    dataset = Echo(split="test", meta_fp="test_view_4_echonet.csv") 
    mean, std = get_mean_and_std(dataset)
    #mean = 1.0
    #std = 1.0
    dataset = Echo(split="test", meta_fp="test_view_4_echonet.csv", mean=mean, std=std)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)   
    model.eval()
    res = []
    y = []
    yhat = []
    for data in tqdm.tqdm(dataloader):
        videos, targets = data
        with torch.no_grad():
            videos.to(device, non_blocking=True)
            y.append(targets.numpy())
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=True):
                preds = model(videos)
                yhat.append(preds.view(-1).cpu().detach().numpy())
                #print(preds.squeeze().shape, targets.shape)
                loss_t = criterion(preds.view(-1), targets)
                #print(loss_t.cpu())
                res.append(loss_t.cpu())
    print("MSE:", np.mean(res), sklearn.metrics.r2_score(np.concatenate(y), np.concatenate(yhat)))

if __name__ == "__main__":
    main()
