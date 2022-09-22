import torch
import numpy as np
import os
import torchvision
from run_echonet import run_epoch
from echonet_dataset import Echo
from dataset_utils import get_mean_and_std
from torch.cuda.amp import autocast
import tqdm
from run_echonet import run_epoch
import neptune.new as neptune
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

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
    num_workers=20
    batch_size=20
    device="cuda"
    path_to_model_weights="/proj/suef_data/echonet_pretrained_weights/"
    num_epochs=45
    lr=1e-4
    weight_decay=1e-4
    lr_step_period=15
    pad=12
    model_save_dir = "/proj/suef_data/saved_models/echonet/"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    ### Init model ###
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    if torch.cuda.is_available():
        print("cuda is available, original weights")
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(path_to_model_weights + "r2plus1d_18_32_2_pretrained.pt")
        model.load_state_dict(checkpoint['state_dict'])
    # Set criterion
    criterion = torch.nn.MSELoss()
    
    ### Init datasets and dataloaders ###
    # Train
    dataset_train = Echo(split="train", meta_fp="train_view_4_echonet.csv") 
    mean, std = get_mean_and_std(dataset_train)
    kwargs = {
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }
    dataset_train = Echo(split="train", meta_fp="train_view_4_echonet.csv", **kwargs, pad=pad)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True)   
    # Val
    dataset_val = Echo(split="val", meta_fp="val_view_4_echonet.csv", **kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    # Test
    #dataset_test = Echo(split="test", meta_fp="test_view_4_echonet.csv", mean=mean, std=std)
    #dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    ### Init Neptune Logging ###
    neptune_tags = []
    neptune_tags.append(model_name)
    neptune_tags.append("echonet_SUEF_data")
    run = neptune.init(project='jenniferalven/lvef', tags=neptune_tags)
    experiment_params = {'weight_decay': weight_decay,
                         'batch_size': batch_size,
                         'num_epochs': num_epochs,
                         'lr_step_period': lr_step_period,
                         'learning_rate' : lr, 
                         'train_dataset_size': len(dataset_train),
                         'val_dataset_size': len(dataset_val)}
    run['parameters'] = experiment_params
    experiment_name = run["sys/id"].fetch()
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    torch.backends.cudnn.benchmark = True
    best_val_loss = 10000 
    for epoch in range(num_epochs):
        t_loss, t_r2_score = run_epoch(model, dataloader_train, train=True, optim=optim, device=device)
        # Log to neptune
        log_train_metrics(run, t_loss, t_r2_score)
        # Run validation step every 5 epoch to save time
        print("Epoch ", epoch, " Train MSE: ", t_loss, " Train R2:", t_r2_score)
        if not epoch % 5:
            v_mse_loss, v_r2_score = run_epoch(model, dataloader_val, train=False, optim=optim, device=device)
            log_val_metrics(run, v_mse_loss, v_r2_score)
            scheduler.step(v_mse_loss)
            if v_mse_loss < best_val_loss:
                checkpoint_name = model_save_dir + model_name + '_ep_' + str(epoch) + '_exp_' + experiment_name
                best_val_loss = v_mse_loss
                torch.save(model.state_dict(), checkpoint_name + '.pth')
                torch.save(optimizer.state_dict(), checkpoint_name + '_op_.pth')
    
def log_train_metrics(experiment, t_loss, t_r2):
    experiment['train/loss'].log(t_loss)
    experiment['train/r2'].log(t_r2)

def log_val_metrics(experiment, val_loss, val_r2):
    experiment['val/loss'].log(val_loss)
    experiment['val/r2'].log(val_r2)

if __name__ == "__main__":
    main()
