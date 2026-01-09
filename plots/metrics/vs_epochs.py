import argparse
import matplotlib.pyplot as plt
from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--metric', type=str)
args = parser.parse_args()

train, val, test = parse_metrics(args.path)
if args.metric is None:
    if 'Mean Absolute Error' in train:
        args.metric = 'Mean Absolute Error'
    else:
        args.metric = 'Cross Entropy Loss'

plt.plot(train[args.metric], label='train')
plt.plot(val[args.metric], label='val')
plt.plot(test[args.metric], label='test')

plt.legend()
plt.grid()
plt.savefig(f"assets/{args.path.replace('/', '-')}.png")