import os
from argparse import Namespace
from datetime import datetime
import pickle
from typing import Union, List, Tuple


def get_time():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class Logger:

    def __init__(self, config: Namespace, others: Union[Namespace, None] = None):

        '''
        Initialize the logging directory:
            ./results/<dataset>/<gnn_layer>/<drop_strategy>/<datetime>/

        Args:
            dataset (str): dataset name.
            model (str): model name.
        '''
        
        self.exp_dir = f'./results/{config.dropout}/{config.dataset}/{config.gnn}/L={len(config.gnn_layer_sizes)}/P={round(config.drop_p, 6)}/{get_time()}'
        os.makedirs(self.exp_dir)
        
        self.log(''.join(f'{k} = {v}\n' for k, v in vars(config).items()), with_time=False)
        with open(f'{self.exp_dir}/config.pkl', 'wb') as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if others is not None:
            self.log(''.join(f'{k} = {v}\n' for k, v in vars(others).items()), with_time=False)
        with open(f'{self.exp_dir}/others.pkl', 'wb') as f:
            pickle.dump(others, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.pickle_dir = None
        self.array_dir = None
        self.tensor_dir = None

    def log(
        self,
        text: str,
        with_time: bool = True,
        print_text: bool = False,
    ):

        '''
        Write logs to the the logging file: ./<exp_dir>/logs

        Args:
            text (str): text to write to the log file.
            with_time (bool): prepend text with datetime of writing.
            print_text (bool): print the text to console, in addition
                to writing it to the log file.
        '''

        if print_text:
            print(text)
        if with_time:
            text = f"{get_time()}: {text}"
        with open(f'{self.exp_dir}/logs', 'a') as f:
            f.write(text + '\n')

    def log_metrics(
        self,
        metrics: List[Tuple[str, float]],
        prefix: str = '',
        with_time: bool = True,
        print_text: bool = False
    ):

        formatted_metrics = prefix
        formatted_metrics += ', '.join(f'{metric} = {value:.6e}' for metric, value in metrics)
        self.log(formatted_metrics, with_time, print_text)