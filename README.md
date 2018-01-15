# Human-Level Control through Deep Reinforcement Learning

This code is the tiny Tensorflow implementation of Deep-Q Network [Human-Level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236).

This code is implemented based on two existing github repos:
- [tiny-dqn](https://github.com/ageron/tiny-dqn) for tiny implementation with Tensorflow
- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) for replay memory, preprocessing, and parameter settings

This implementation contains:
- Deep Q-network and Q-learning
- Random start game
- Experience replay memory
    - to reduce the correlations between consecutive updates
- Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values
- Use Huber loss instead of clipping the gradients of mean-squared-error (MSE) loss (different from the paper)
    - to improve the stability of training

So far, I only tested this code with the `Breakout-v0`.

## Environment
```
atari-py (0.1.1)
backports.weakref (1.0.post1)
bleach (1.5.0)
Box2D-kengz (2.3.3)
certifi (2017.11.5)
chardet (3.0.4)
enum34 (1.1.6)
funcsigs (1.0.2)
future (0.16.0)
futures (3.1.1)
gym (0.7.0)
html5lib (0.9999999)
idna (2.6)
imageio (2.2.0)
Keras (2.1.1)
Markdown (2.6.9)
mock (2.0.0)
mujoco-py (0.5.7)
numpy (1.13.3)
olefile (0.44)
opencv-python (3.3.0.10)
pachi-py (0.0.21)
pbr (3.1.1)
Pillow (4.3.0)
pip (9.0.1)
protobuf (3.5.0.post1)
pyglet (1.3.0)
PyOpenGL (3.1.0)
PyYAML (3.12)
requests (2.18.4)
scipy (1.0.0)
setuptools (37.0.0)
six (1.11.0)
tensorflow-gpu (1.4.0)
tensorflow-tensorboard (0.4.0rc3)
Theano (1.0.0)
tqdm (4.19.4)
urllib3 (1.22)
Werkzeug (0.12.2)
wheel (0.30.0)
```


## Training
```
python main.py -v
```


## Testing
```
python main.py --test --render
```


## License
- For academic and non-commercial use only
- Apache License 2.0
