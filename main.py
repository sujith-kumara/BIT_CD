import os
import torch
from models.trainer import CDTrainer
from models.evaluator import CDEvaluator
import utils

def train():
    args = {
        'gpu_ids': '0', 
        'project_name': 'test', 
        'checkpoint_root': 'checkpoints',
        'num_workers': 4, 
        'dataset': 'CDDataset', 
        'data_name': 'LEVIR',
        'batch_size': 8,
        'split': "train",
        'split_val': "val",
        'img_size': 256,
        'n_class': 2,
        'net_G': 'base_transformer_pos_s4_dd8',
        'loss': 'ce',
        'optimizer': 'sgd',
        'lr': 0.01,
        'max_epochs': 100,
        'lr_policy': 'linear',
        'lr_decay_iters': 100
    }

    # Initialize device
    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create checkpoint and visualization directories
    args['checkpoint_dir'] = os.path.join(args['checkpoint_root'], args['project_name'])
    os.makedirs(args['checkpoint_dir'], exist_ok=True)
    args['vis_dir'] = os.path.join('vis', args['project_name'])
    os.makedirs(args['vis_dir'], exist_ok=True)

    # Get data loaders
    dataloaders = utils.get_loaders(args)

    # Train model
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test():
    args = {
        'data_name': 'LEVIR',
        'img_size': 256,
        'batch_size': 8,
        'split_val': 'test',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Get data loader for testing
    dataloader = utils.get_loader(args['data_name'], img_size=args['img_size'],
                                  batch_size=args['batch_size'], is_train=False,
                                  split=args['split_val'])

    # Test model
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()

if __name__ == '__main__':
    train()
    test()
