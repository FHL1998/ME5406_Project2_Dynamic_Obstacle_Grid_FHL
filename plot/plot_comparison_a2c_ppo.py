import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas


def mean_smooth(input_data):
    return pandas.Series(list(input_data)).rolling(args.smoothness, min_periods=5).mean()


parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='return')
parser.add_argument('--save', type=str, default='comparison_return')
parser.add_argument('--smoothness', type=int, default=1000)
args = parser.parse_args()

plt.rc('font', family='Times New Roman', size=12)
f1, ax1 = plt.subplots(1, 1)
csv_ppo = pandas.read_csv('csvs/ppo.csv')
csv_a2c = pandas.read_csv('csvs/a2c.csv')

ax1.plot(csv_ppo['frames'], mean_smooth(csv_ppo['return_mean']), color='green', label='PPO')
ax1.plot(csv_a2c['frames'], mean_smooth(csv_a2c['return_mean']), color='purple', label='A2C')

ax1.fill_between(csv_ppo['frames'], mean_smooth((csv_ppo['return_mean'] - csv_ppo['return_std'])),
                 mean_smooth((csv_ppo['return_mean'] + csv_ppo['return_std'])),
                 color='green', alpha=0.2)

ax1.fill_between(csv_a2c['frames'], mean_smooth((csv_a2c['return_mean'] - csv_a2c['return_std'])),
                 mean_smooth((csv_a2c['return_mean'] + csv_a2c['return_std'])),
                 color='purple', alpha=0.2)

ax1.set_title('Comparison Metric: Return')
ax1.legend(loc='right')
ax1.set_xlabel('Update Frames')
ax1.set_ylabel('Return')

f2, ax2 = plt.subplots(1, 1)
ax2.plot(csv_ppo['frames'], csv_ppo['entropy'], color='green', alpha=0.2)
ax2.plot(csv_a2c['frames'], csv_a2c['entropy'], color='purple', alpha=0.2)
ax2.plot(csv_ppo['frames'], mean_smooth(csv_ppo['entropy']), color='green', label='PPO: Entropy Loss')
ax2.plot(csv_a2c['frames'], mean_smooth(csv_a2c['entropy']), color='purple', label='A2C: Entropy Loss')
# ax2.plot(csv_ppo['frames'], csv_ppo['policy_loss'], color='blue', alpha=0.2)
# ax2.plot(csv_a2c['frames'], csv_a2c['policy_loss'], color='red', alpha=0.2)
# ax2.plot(csv_ppo['frames'], mean_smooth(csv_ppo['policy_loss']), color='blue', label='PPO: Policy Loss')
# ax2.plot(csv_a2c['frames'], mean_smooth(csv_a2c['policy_loss']), color='red', label='A2C: Policy Loss')

ax2.set_title('Comparison Metric: Loss')
ax2.legend(loc='right')
ax2.set_xlabel('Update Frames')
ax2.set_ylabel('Entropy Loss')

f3, ax3 = plt.subplots(1, 1)
ax3.plot(csv_ppo['frames'], csv_ppo['value'], color='green', alpha=0.2)
ax3.plot(csv_a2c['frames'], csv_a2c['value'], color='purple', alpha=0.2)
ax3.plot(csv_ppo['frames'], mean_smooth(csv_ppo['value']), color='green', label='PPO')
ax3.plot(csv_a2c['frames'], mean_smooth(csv_a2c['value']), color='purple', label='A2C')
ax3.set_title('Comparison Metric: Value')
ax3.legend(loc='right')
ax3.set_xlabel('Update Frames')
ax3.set_ylabel('Value')

# plt.show()


if args.save:
    os.makedirs('plots', exist_ok=True)
    f1.savefig(os.path.join('plots', args.save + '.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    f2.savefig(os.path.join('plots', 'compare_entropy' + '.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    f3.savefig(os.path.join('plots', 'compare_value' + '.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)
else:
    plt.show()
