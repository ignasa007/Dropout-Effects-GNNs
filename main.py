from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataset, BaseDataset
from model import Model
from utils.config import parse_arguments
from utils.logger import Logger, sci_notation
from utils.format import format_task_name, FormatEpoch


config, others = parse_arguments(return_others=True)
DEVICE = torch.device(f'cuda:{config.device_index}' if torch.cuda.is_available() and config.device_index is not None else 'cpu')

dataset: BaseDataset = get_dataset(config.dataset, config=config, others=others, device=DEVICE)
others.input_dim = dataset.num_features
others.output_dim = dataset.output_dim
others.task_name = format_task_name.get(dataset.task_name.lower())

model = Model(config, others)
if config.dataset.lower().startswith('SyntheticZINC'.lower()) and others.model_sample is not None:
    load_fn = f'./results/synthetic-zinc_state-dicts/{config.gnn}/sample-{others.model_sample}.pt'
    state_dict = torch.load(load_fn, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    print(f'Successfully loaded state-dict {load_fn}.')
model = model.to(DEVICE)

lr = config.learning_rate
optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
if config.schedule_lr:
    scheduling_metric = 'Cross Entropy Loss' if dataset.task_name.lower().endswith('-c') else 'Mean Absolute Error'
    scheduler = ReduceLROnPlateau(
        optimizer, patience=10//config.test_every, min_lr=1e-8,
        # Default arguments, replicating FoSR (Karhadkar et al., 2022)
        # https://github.com/kedar2/FoSR/blob/1a7360c2c77c42624bdc7ffef1490a2eb0a8afd0/experiments/graph_classification.py#L78
        mode='min', factor=0.1, threshold=1e-4
    )

logger = Logger(config, others)
format_epoch = FormatEpoch(config.n_epochs)

for epoch in tqdm(range(1, config.n_epochs+1)):

    logger.log(f'Epoch {format_epoch(epoch)}', with_time=True)
    train_metrics = dataset.train(model, optimizer)
    logger.log_metrics(train_metrics, prefix='\tTraining:'.ljust(13), with_time=False)

    if epoch == config.n_epochs or config.test_every > 0 and epoch % config.test_every == 0:

        val_metrics, test_metrics = dataset.eval(model)
        logger.log_metrics(val_metrics, prefix='\tValidation:'.ljust(13), with_time=False)
        logger.log_metrics(test_metrics, prefix='\tTesting:'.ljust(13), with_time=False)
        
        if config.schedule_lr:
            scheduler.step([value for metric, value in val_metrics if metric == scheduling_metric][0])
            if lr != optimizer.param_groups[0]['lr']:
                logger.log(f"\tUpdated learning rate from {sci_notation(lr, decimals=6, strip=True)} to {sci_notation(optimizer.param_groups[0]['lr'], decimals=6, strip=True)}.", with_time=False)
                lr = optimizer.param_groups[0]['lr']

    if isinstance(config.save_every, int) and (config.save_every > 0 and epoch % config.save_every == 0 or config.save_every == -1 and epoch == config.n_epochs):
        ckpt_fn = f'{logger.exp_dir}/ckpt-{format_epoch(epoch)}.pt'
        logger.log(f'\tSaving model at {ckpt_fn}.', with_time=False)
        torch.save(model.state_dict(), ckpt_fn) 

    logger.log('', with_time=False)