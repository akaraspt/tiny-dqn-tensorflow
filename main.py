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

from history import History
from replay_memory import ReplayMemory
from utils import rgb2gray, imresize

# env = gym.make("MsPacman-v0")
env = gym.make("Breakout-v0")
done = True  # env needs to be reset

# First let's build the two DQNs (online & target)
# input_height = 88
# input_width = 80
# input_channels = 1
input_height = 84
input_width = 84
input_channels = history_length = 4
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
# conv_paddings = ["SAME"] * 3
conv_paddings = ["VALID"] * 3
conv_activation = [tf.nn.relu] * 3
# n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
# initializer = tf.contrib.layers.variance_scaling_initializer()
initializer = tf.truncated_normal_initializer(0, 0.02)

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

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# Now for the training operations
# learning_rate = 0.001
learning_rate = 0.00025
momentum = 0.95

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    # y = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None])

    # q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
    #                         axis=1, keep_dims=True)
    # error = tf.abs(y - q_value)
    # clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    # linear_error = 2 * (error - clipped_error)
    # loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
    
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1)
    delta = y - q_value
    loss = tf.reduce_mean(clipped_error(delta))

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# # Let's implement a simple replay memory
# replay_memory_size = 20000
# replay_memory = deque([], maxlen=replay_memory_size)

# def sample_memories(batch_size):
#     indices = np.random.permutation(len(replay_memory))[:batch_size]
#     cols = [[], [], [], [], []] # state, action, reward, next_state, continue
#     for idx in indices:
#         memory = replay_memory[idx]
#         for col, value in zip(cols, memory):
#             col.append(value)
#     cols = [np.array(col) for col in cols]
#     return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
#            cols[4].reshape(-1, 1))

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.1
eps_max = 1.0 if not args.test else eps_min
eps_decay_steps = args.number_steps // 2

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

# # We need to preprocess the images to speed up training
# mspacman_color = np.array([210, 164, 74]).mean()

# def preprocess_observation(obs):
#     img = obs[1:176:2, ::2] # crop and downsize
#     img = img.mean(axis=2) # to greyscale
#     img[img==mspacman_color] = 0 # Improve contrast
#     img = (img - 128) / 128 - 1 # normalize from -1. to 1.
#     return img.reshape(88, 80, 1)
def preprocess_observation(obs):
    return imresize(rgb2gray(obs)/255., (input_width, input_height))

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

history = History(
    data_format='NHWC',
    batch_size=batch_size,
    history_length=history_length,
    screen_height=input_height,
    screen_width=input_width
)

replay_memory_size = 20000
replay_memory = ReplayMemory(
    data_format='NHWC',
    batch_size=batch_size,
    history_length=history_length,
    screen_height=input_height,
    screen_width=input_width,
    memory_size=replay_memory_size,
    model_dir='model'
)

with tf.Session() as sess:
    # if os.path.isfile(args.path + ".index"):
    #     saver.restore(sess, args.path)
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
    while True:
        step = global_step.eval()
        if step >= args.number_steps:
            break
        iteration += 1
        # if args.verbosity > 0:
        #     print("\rIteration {}   Training step {}/{} ({:.1f})%   "
        #           "Loss {:5f}    Mean Max-Q {:5f}   ".format(
        #     iteration, step, args.number_steps, step * 100 / args.number_steps,
        #     loss_val, mean_max_q), end="")
        if args.verbosity > 0:
            print("\rIter {}, training step {}/{} ({:.1f})%, "
                  "loss {:5f}, exp-moving-avg reward {:5f}, "
                  "mean max-Q {:5f}   ".format(
            iteration, step, args.number_steps, step * 100 / args.number_steps,
            loss_val, exp_moving_avg_reward, 
            mean_max_q), end="")
        if done: # game over, start again
            obs = env.reset()
            # for skip in range(skip_start): # skip the start of each game
            for skip in xrange(random.randint(0, skip_start - 1)):
                obs, _, done, _ = env.step(0)
            state = preprocess_observation(obs)
            for _ in range(history_length):
                history.add(state)

        if args.render:
            env.render()

        # Online DQN evaluates what to do
        # q_values = online_q_values.eval(feed_dict={X_state: [state]})
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
        # replay_memory.append((state, action, reward, next_state, 1.0 - done))
        replay_memory.add(next_state, reward, action, done)
        state = next_state

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
        # X_state_val, X_action_val, rewards, X_next_state_val, continues = (
        #     sample_memories(batch_size))
        X_state_val, X_action_val, rewards, X_next_state_val, terminal = \
             replay_memory.sample()
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        # max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        # y_val = rewards + continues * discount_rate * max_next_q_values
        max_next_q_values = np.max(next_q_values, axis=1)
        y_val = rewards + (1. - terminal) * discount_rate * max_next_q_values

        if first_train_step:
            exp_moving_avg_reward = np.mean(rewards)
            first_train_step = False
        else:
            exp_moving_avg_reward = (exp_moving_avg_reward * 0.99) + (0.01 * np.mean(rewards))

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            print("Copying the weight from online DQN to target DQN ...")
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            print("Saving model ...")
            saver.save(sess, os.path.join(args.path, args.model_fname), global_step=step)
