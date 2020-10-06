import pandas as pd
import math
from sklearn.metrics import r2_score, mean_squared_error
import hydra
from omegaconf import DictConfig, OmegaConf
import neptune


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    views = [2, 4]
    suffix = ''
    for view in views:
        suffix = suffix + str(view) + '_'
    names = ['3c', '4c']
    view_results = []
    base = '/home/ola/Projects/SUEF/results/'
    column_list = []

    for view, name in zip(views, names):
        pd_img = pd.read_csv(base + 'inter_' + name + '_' + suffix + 'img.csv', sep=';')
        pd_flow = pd.read_csv(base + 'inter_' + name + '_' + suffix + 'flow.csv', sep=';')
        pd_2stream = pd_img.merge(pd_flow, on=['us_id', 'target'], how='inner')
        c_name = 'avg_pred_' + str(view)
        column_list.append(c_name)
        pd_2stream[c_name] = pd_2stream[['pred_x', 'pred_y']].mean(axis=1)
        view_results.append(pd_2stream[['us_id', 'target', c_name]])

    for vr, c, name in zip(view_results, column_list, names):
        preds = vr[c].to_numpy()
        targets = vr['target'].to_numpy()
        r2 = r2_score(targets, preds)
        mse = mean_squared_error(targets, preds)
        r = math.sqrt(r2)

        print("For view {}, R is {} and MSE is: {}".format(name, r, mse))

    merged = view_results[0].merge(view_results[1], on=['us_id', 'target'], how='inner')

    for i in range(2, len(view_results)):
        merged = merged.merge(view_results[i], on=['us_id', 'target'], how='inner')

    merged['avg_pred_all_views'] = merged[column_list].mean(axis=1)
    preds = merged['avg_pred_all_views'].to_numpy()
    targets = merged['target'].to_numpy()
    print(len(targets))
    r2 = r2_score(targets, preds)
    r = math.sqrt(r2)
    mse = mean_squared_error(targets, preds)
    print("For {} views, R is {} and MSE is: {}".format(len(views), r, mse))
    '''
    neptune.init('eiphodos/SUEF')
    experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                         **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                         'data_stream': '4stream', 'view': '3-Chamber, 4-Chamber'}
    neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params)
    neptune.log_metric('loss', mse)
    neptune.log_metric('r2', r2)
    '''
if __name__ == "__main__":
    main()