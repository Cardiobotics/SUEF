import torch
import torch.nn as nn
from utils import AverageMeter
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
import csv_dataset
from models import linear_model


def main():

    flow_3c_data = '/home/ola/Projects/SUEF/results/train_inter_3c_2_4_flow.csv'
    img_3c_data = '/home/ola/Projects/SUEF/results/train_inter_3c_2_4_img.csv'
    flow_4c_data = '/home/ola/Projects/SUEF/results/train_inter_4c_2_4_flow.csv'
    img_4c_data = '/home/ola/Projects/SUEF/results/train_inter_4c_2_4_img.csv'

    data_set = csv_dataset.CSVDataset(flow_3c_data, img_3c_data, flow_4c_data, img_4c_data)
    data_loader = DataLoader(data_set, batch_size=512, num_workers=10, drop_last=True, shuffle=True)

    v_flow_3c_data = '/home/ola/Projects/SUEF/results/inter_3c_2_4_flow.csv'
    v_img_3c_data = '/home/ola/Projects/SUEF/results/inter_3c_2_4_img.csv'
    v_flow_4c_data = '/home/ola/Projects/SUEF/results/inter_4c_2_4_flow.csv'
    v_img_4c_data = '/home/ola/Projects/SUEF/results/inter_4c_2_4_img.csv'

    linear_model_path = '/home/ola/Projects/SUEF/saved_models/best_linear.pth'

    v_data_set = csv_dataset.CSVDataset(v_flow_3c_data, v_img_3c_data, v_flow_4c_data, v_img_4c_data)
    v_data_loader = DataLoader(v_data_set, batch_size=64, num_workers=10, drop_last=True, shuffle=True)

    model = linear_model.MultiViewLinear(4)
    model.to('cuda')
    torch.backends.cudnn.benchmark = True

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=2.2)

    losses_t = AverageMeter()
    r2_values_t = AverageMeter()

    losses_v = AverageMeter()
    r2_values_v = AverageMeter()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    best_r2 = 0

    weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], device='cuda:0', requires_grad=True)
    model.fc_linear.weight.data = weights

    print("Val dataset length: {}".format(len(v_data_set)))

    for i in range(500):
        model.train()
        for inputs, targets in data_loader:
            targets = targets.to('cuda', non_blocking=True)
            inputs = inputs.to('cuda', non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses_t.update(loss)

            t_r2 = r2_score(targets.cpu().detach(), outputs.cpu().detach())
            r2_values_t.update(t_r2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print("Epoch {} | Average Training Loss: {} Average R2 Score: {}".format(i+1, losses_t.avg, r2_values_t.avg))

        model.eval()
        for v_inputs, v_targets in v_data_loader:
            v_targets = v_targets.to('cuda', non_blocking=True)
            v_inputs = v_inputs.to('cuda', non_blocking=True)

            with torch.no_grad():
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_targets)

            losses_v.update(v_loss)

            v_r2 = r2_score(v_targets.cpu().detach(), v_outputs.cpu().detach())
            r2_values_v.update(v_r2)

        print("Epoch {} | Average Validation Loss: {} Average R2 Score: {}".format(i + 1, losses_v.avg, r2_values_v.avg))
        if r2_values_v.avg > best_r2:
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, linear_model_path)
            best_r2 = r2_values_v.avg

    model = linear_model.MultiViewLinear(4)
    model.to('cuda')
    model.load_state_dict(torch.load(linear_model_path))

    print("Best R2: {}".format(best_r2))
    print("Weights: {}".format(model.fc_linear.weight))
    print("Bias: {}".format(model.fc_linear.bias))


if __name__ == "__main__":
    main()