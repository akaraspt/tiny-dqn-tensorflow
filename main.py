from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net to play MsMacman.")
parser.add_argument("-n", "--number-steps", type=int, default=4000000,
    help="total number of training steps")
parser.add_argument("-l", "--learn-iterations", type=int, default=4,
    help="number of game iterations between each training step")
parser.add_argument("-s", "--save-steps", type=int, default=1000,
    help="number of training steps between saving checkpoints")
parser.add_argument("-c", "--copy-steps", type=int, default=10000,
    help="number of training steps between copies of online DQN to target DQN")
parser.add_argument("-r", "--render", action="store_true", default=False,
    help="render the game during training or testing")
parser.add_argument("-p", "--path", default="model",
    help="path of the checkpoint file")
parser.add_argument("-m", "--model_fname", default="model.ckpt",
    help="name of the checkpoint file")
parser.add_argument("-t", "--test", action="store_true", default=False,
    help="test (no learning and minimal epsilon)")
parser.add_argument("-v", "--verbosity", action="count", default=0,
    help="increase output verbosity")
args = parser.parse_args()

from collections import deque
import gym
import numpy as np
import os
import tensorflow as tf
import random
import sys

from history import History
from replay_memory import ReplayMemory
from utils import rgb2gray, imresize

env = gym.make("Breakout-v0")
done = True  # env needs to be reset

# First let's build the two DQNs (online & target)
input_height = 84
input_width = 84
input_channels = history_length = 4
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["VALID"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.truncated_normal_initializer(0, 0.02)

# Deep-Q network
def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        prev_layer_shape = prev_layer.get_shape().as_list()
        n_hidden_in = reduce(lambda x, y: x * y, prev_layer_shape[1:])
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

# Place holder for input
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])

# Create two Deep-Q networks
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# Parameters for optimizer
learning_rate = 0.00025
learning_rate_minimum = 0.00025
learning_rate_decay = 0.96
learning_rate_decay_step = 50000
momentum = 0.95

# Huber loss
def clipped_error(x):
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

# Initialize optimizer for training
with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])   # Action based on Q-value from Online network
    y = tf.placeholder(tf.float32, shape=[None])        # Q-value from Target network

    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1)
    delta = y - q_value
    loss = tf.reduce_mean(clipped_error(delta))

    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
    learning_rate_op = tf.maximum(
        learning_rate_minimum,
        tf.train.exponential_decay(
            learning_rate,
            learning_rate_step,
            learning_rate_decay_step,
            learning_rate_decay,
            staircase=True
        )
    )
    training_op = tf.train.RMSPropOptimizer(
        learning_rate_op, momentum=momentum, epsilon=0.01
    ).minimize(loss, global_step=global_step)

# Summary for Tensorboard
summary_steps = 100
with tf.variable_scope('summary'):
    summary_tags = ['average.reward']

    summary_placeholders = {}
    summary_ops = {}

    for tag in summary_tags:
        summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        summary_ops[tag]  = tf.summary.scalar(tag, summary_placeholders[tag])

    # histogram_summary_tags = ['episode.rewards', 'episode.actions']

    # for tag in histogram_summary_tags:
    #     summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
    #     summary_ops[tag]  = tf.summary.histogram(tag, summary_placeholders[tag])

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# TensorFlow - Execution phase
training_start = 10000  # start training after 10,000 game iterations
discount_rate = 0.99
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 32
iteration = 0  # game iterations
done = True # env needs to be reset
min_reward = -1.
max_reward = 1.
exp_moving_avg_reward = 0.
first_train_step = True

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

# Initialize history
history = History(
    data_format='NHWC',
    batch_size=batch_size,
    history_length=history_length,
    screen_height=input_height,
    screen_width=input_width
)

# Initialize Replay Memory
replay_memory_size = 1000000
replay_memory = ReplayMemory(
    data_format='NHWC',
    batch_size=batch_size,
    history_length=history_length,
    screen_height=input_height,
    screen_width=input_width,
    memory_size=replay_memory_size,
    model_dir='model'
)

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.1
eps_max = 1.0
test_eps = eps_min if args.test else None
eps_decay_steps = 1000000
# eps_min = 0.1
# eps_max = 1.0 if not args.test else eps_min
# eps_decay_steps = args.number_steps // 2

def epsilon_greedy(q_values, step):
    epsilon = test_eps or \
              (
                eps_min + max(
                    0., 
                    (eps_max - eps_min) * (eps_decay_steps - max(0., step - training_start)) / eps_decay_steps
                )
              )
    # epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

def preprocess_observation(obs):
    return imresize(rgb2gray(obs)/255., (input_width, input_height))

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(args.path, "logs"), sess.graph)

    def inject_summary(tag_dict, step):
        summary_str_lists = sess.run(
            [summary_ops[tag] for tag in tag_dict.keys()],
            {summary_placeholders[tag]: value for tag, value in tag_dict.items()}
        )
        for summary_str in summary_str_lists:
            summary_writer.add_summary(summary_str, step)

    # Resume the training (if possible)
    ckpt = tf.train.get_checkpoint_state(args.path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        fname = os.path.join(args.path, ckpt_name)
        saver.restore(sess, fname)
        print(" [*] Load SUCCESS: %s" % fname)
    else:
        init.run()
        copy_online_to_target.run()
        print(" [!] Load FAILED: %s" % args.path)

    # Training
    exp_moving_avg_reward = 0.0
    current_rewards = []
    while True:
        step = global_step.eval()
        if step >= args.number_steps:
            break
        iteration += 1
        if args.verbosity > 0:
            print("\rIter {}, training step {}/{} ({:.1f})%, "
                  "loss {:5f}, exp-moving-avg reward {:5f}, "
                  "mean max-Q {:5f}".format(
                    iteration, step, args.number_steps, step * 100 / args.number_steps,
                    loss_val, exp_moving_avg_reward, 
                    mean_max_q), 
                end=""
            )

        # Game over, start again
        if done:
            obs = env.reset()

            # Randomly skip the start of each game
            for skip in xrange(random.randint(0, skip_start - 1)):
                obs, _, done, _ = env.step(0)

            state = preprocess_observation(obs)
            for _ in range(history_length):
                history.add(state)

        if args.render:
            env.render()

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [history.get()]})[0]
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Reward clipping
        reward = max(min_reward, min(max_reward, reward))

        # Update history
        history.add(next_state)

        # Let's memorize what happened
        replay_memory.add(next_state, reward, action, done)
        state = next_state
        current_rewards.append(reward)

        if args.test:
            continue

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % args.learn_iterations != 0:
            continue # only train after warmup period and at regular intervals

        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, terminal = \
             replay_memory.sample()
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val}
        )
        max_next_q_values = np.max(next_q_values, axis=1)
        y_val = rewards + (1. - terminal) * discount_rate * max_next_q_values

        # Update exponential moving average of rewards
        if first_train_step:
            exp_moving_avg_reward = np.mean(current_rewards)
            first_train_step = False
        else:
            exp_moving_avg_reward = (exp_moving_avg_reward * 0.99) + (0.01 * np.mean(current_rewards))
        current_rewards = []

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val,
            X_action: X_action_val,
            y: y_val,
            learning_rate_step: step,
        })

        # Regularly inject summary
        if step % summary_steps == 0:
            inject_summary(
                {
                    'average.reward': exp_moving_avg_reward
                }, 
                step
            )

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            # print("Copying the weight from online DQN to target DQN ...")
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            # print("Saving model ...")
            saver.save(sess, os.path.join(args.path, args.model_fname), global_step=step)
