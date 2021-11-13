import matplotlib.pyplot as plt
import pandas


# frame = pandas.read_csv('results/a2c_1_0.001/log.csv', engine='python')
# data = frame.drop_duplicates(subset=['update'], keep='first', inplace=False)
# data.to_csv('csvs/a2c_1_0.001.csv', encoding='utf8', index=False)

def mean_smooth(input_data):
    return pandas.Series(list(input_data)).rolling(2000, min_periods=5).mean()


csv_a2c_lr_1 = pandas.read_csv('results/csvs/a2c_1_0.001.csv')
csv_a2c_lr_2 = pandas.read_csv('results/csvs/a2c_1_0.00085.csv')

csv_ppo_clip_1 = pandas.read_csv('results/csvs/ppo_clip_0.1.csv')
csv_ppo = pandas.read_csv('results/csvs/ppo.csv')
csv_ppo_clip_4 = pandas.read_csv('results/csvs/ppo_clip_0.4.csv')

plt.rc('font', family='Times New Roman', size=12)
f, ax = plt.subplots(1, 1)

# ax.plot(csv_ppo_clip_1['frames'], mean_smooth(csv_ppo_clip_1['value']), color='navy', label='PPO clip epsilon:0.1')
# ax.plot(csv_ppo_clip_1['frames'], csv_ppo_clip_1['value'], color='navy', alpha=0.15)
# ax.plot(csv_ppo['frames'], mean_smooth(csv_ppo['value']), color='blue', label='PPO clip epsilon:0.2')
# ax.plot(csv_ppo['frames'], csv_ppo['value'], color='blue', alpha=0.15)
# ax.plot(csv_ppo_clip_4['frames'], mean_smooth(csv_ppo_clip_4['value']), color='cadetblue', label='PPO clip epsilon:0.4')
# ax.plot(csv_ppo_clip_4['frames'], csv_ppo_clip_4['value'], color='cadetblue', alpha=0.15)

ax.plot(csv_a2c_lr_1['frames'], mean_smooth(csv_a2c_lr_1['return_mean']), color='green',
        label='A2C learning rate:0.001 (max. mean return: %.1f)' % max(csv_a2c_lr_1['return_mean']))
ax.fill_between(csv_a2c_lr_1['frames'], mean_smooth((csv_a2c_lr_1['return_mean'] - csv_a2c_lr_1['return_std'])),
                mean_smooth((csv_a2c_lr_1['return_mean'] + csv_a2c_lr_1['return_std'])),
                color='green', alpha=0.15)

ax.plot(csv_a2c_lr_2['frames'], mean_smooth(csv_a2c_lr_2['return_mean']), color='darkorange',
        label='A2C learning rate:0.00085 (max. mean return: %.1f)' % max(csv_a2c_lr_2['return_mean']))
ax.fill_between(csv_a2c_lr_2['frames'], mean_smooth((csv_a2c_lr_2['return_mean'] - csv_a2c_lr_2['return_std'])),
                mean_smooth((csv_a2c_lr_2['return_mean'] + csv_a2c_lr_2['return_std'])),
                color='darkorange', alpha=0.15)

# ax.set_title('PPO Tuning Clip Epsilon Metric: Value')
# ax.legend(loc='best')
# ax.set_xlabel('Update Frames')
# ax.set_ylabel('Value')
# plt.show()
# f.savefig('plots/PPO/ppo_tuning_clip_epsilon.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)

ax.set_title('A2C Tuning Learning Rate Metric: Return')
ax.legend(loc='best')
ax.set_xlabel('Update Frames')
ax.set_ylabel('Return')
plt.show()
f.savefig('plots/A2C/a2c_tuning_learning_rate.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
