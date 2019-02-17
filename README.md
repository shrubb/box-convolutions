Box Convolution Layer for ConvNets
==================================

<p align="center">
<img src="https://user-images.githubusercontent.com/9570420/52168173-d1ea7980-2737-11e9-8924-e4e8fd39d0ee.gif">
<br>
Single-box-conv network (from `test-mnist.py`) learns patterns on MNIST
</p>

# What This Is

This is a PyTorch implementation of the box convolution layer as introduced in the 2018 NeurIPS [paper](https://papers.nips.cc/paper/7859-deep-neural-networks-with-box-convolutions):

Burkov, E., & Lempitsky, V. (2018) **Deep Neural Networks with Box Convolutions**. *Advances in Neural Information Processing Systems 31*, 6214-6224.

# How to Use

## Installing

```bash
git clone https://github.com/shrubb/box-convolutions.git && cd box-convolutions
pip3 install --user .
python3 -m box_convolution.test # if throws errors, please open a GitHub issue
```

To uninstall:

```bash
pip3 uninstall box_convolution
```

## Using

```python3
import torch
from box_convolution import BoxConv2d

box_conv = BoxConv2d(16, 8, 240, 320)
help(BoxConv2d)
```

Also, there is a usage example in `test-mnist.py`.

Tested on Ubuntu 18.04.2, Python 3.6, PyTorch 1.0.0, GCC 7.3.0, CUDA 9.2. Other versions (e.g. Python 2.7 or PyTorch 0.4 or GCC 5 or CUDA 8 or CUDA 10) should work too, but I haven't checked. If something doesn't build, please open an issue.

CUDA 9/9.1 + GCC 6 isn't supported, see [this issue](https://github.com/shrubb/box-convolutions/issues/2).

# Quick Tour of Box convolutions

You may want to see our [poster](https://yadi.sk/i/LNnMrj6FwbOc9A).

### Why reinvent the old convolution?

`3×3` convolutions are too small ⮕ receptive field grows too slow ⮕ ConvNets have to be very deep.

This is especially undesirable in dense prediction tasks (*segmentation, depth estimation, object detection, ...*).

Today people solve this by

* dilated/deformable convolutions (can bring artifacts or degrade to `1×1` conv; almost always filter high-frequency);
* "global" spatial pooling layers (usually too constrained, fixed size, not "fully convolutional").

### How does it work?

Box convolution layer is a basic *depthwise convolution* (i.e. `Conv2d` with `groups=in_channels`) but with special kernels called *box kernels*.

A box kernel is a rectangular averaging filter. That is, filter values are fixed and unit! Instead, we learn four parameters per rectangle − its size and offset:

![image](https://user-images.githubusercontent.com/9570420/41361143-f6db467a-6f36-11e8-9dfc-086a79256bfc.png)

![image](https://user-images.githubusercontent.com/9570420/40393137-f371e1ea-5e26-11e8-868a-79ea3f6847f1.png)

### Any success stories?

One example: there is an efficient semantic segmentation model [**ENet**](https://github.com/e-lab/ENet-training). It's a classical hourglass architecture stacked of dozens ResNet-like blocks (left image).

Let's replace some of these blocks by our "box convolution block" (right image).

<img src="https://user-images.githubusercontent.com/9570420/50013966-a9fe5580-ffd3-11e8-8824-8b1b1673ba83.png" width="530">

First we replaced every second block with a box convolution block (*Box*ENet in the paper). The model became

* more accurate,
* faster,
* lighter
* **without dilated convolutions**.

Then, we replaced **every** residual block (except the down- and up-sampling ones)! The result, *BoxOnly*ENet, is

* a **ConvNet almost without** (traditional learnable weight) **convolutions**,
* **2** times less operations,
* **3** times less parameters,
* still **more accurate** than ENet!

