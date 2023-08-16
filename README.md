# Automatic Image Colorization

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Template from jarvis](https://img.shields.io/badge/Hi-Jarvis-ff69b4.svg)](https://github.com/Armour/Jarvis)

## Overview

This is a TensorFlow implementation of the Residual Encoder Network based on [Automatic Colorization](http://tinyclouds.org/colorize/) and the pre-trained VGG16 model from [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

**For latest TensorFlow with [estimator](https://www.tensorflow.org/guide/estimators) support, check [tf-1.12](https://github.com/Armour/Automatic-Image-Colorization/tree/tf-1.12) branch. (still under development, the training code is working now)**

## Structure

* `config.py`: config variables like batch size, training_iters and so on
* `image_helper.py`: all functions related to image manipulation
* `read_input.py`: all functions related to input
* `residual_encoder.py`: the residual encoder model
* `common.py`: the common part for training and testing, which is mainly the workflow for this model
* `train.py`: train the residual encoder model using TensorFlow built-in AdamOptimizer
* `test.py`: test your own images and save the output images

## TensorFlow graph

![residual_encoder](residual_encoder.png)

## How to use

* Download pre-trained VGG16 model [vgg16.npy](https://drive.google.com/u/0/uc?id=1d0BRPQomC40L5xClfmuUBayRdJIdqKVb&export=download&confirm=t&uuid=ffb6f7e1-d71f-4f5d-a6a5-083f0b767e54&at=AB6BwCDjgV_wW8zsEGwQE5_Oh9kz:1691880071525) to vgg folder

* Option 1: Use a pre-trained residual encoder model
  * Download model [here](https://github.com/git-preet/IMAGE_COLORIZATION)
  * Unzip all files to `summary_path` (you can change this path in `config.py`)

* Option 2: Train your own model!
  1. Change the `batch_size` and `training_iters` if you want.
  2. Change `training_dir` to your directory that has all your training jpg images
  3. Run `python train.py`

* Test
  1. Change `testing_dir` to your directory that has all your testing jpg images
  2. Run `python test.py`

## Examples

* ![1](1.png)
* ![2](2.png)
* ![3](3.png)
* ![4](4.png)
* ![5](5.png)
* ![6](6.png)
* ![7](7.png)
* ![8](8.png)
* ![9](9.png)
* ![10](10.png)
* ![11](11.png)
* ![12](12.png)

* More example output images can be found in [sample_output_images](https://github.com/git-preet/IMAGE_COLORIZATION) folder.

## References

* [Automatic Colorization](http://tinyclouds.org/colorize/)
* [pavelgonchar/colornet](https://github.com/pavelgonchar/colornet)
* [raghavgupta0296/ColourNet](https://github.com/raghavgupta0296/ColourNet)
* [pretrained VGG16 npy file](https://github.com/machrisaa/tensorflow-vgg)


## License

[GNU GPL 3.0](https://github.com/git-preet/IMAGE_COLORIZATION/blob/main/LICENSE) for personal or research use. COMMERCIAL USE PROHIBITED.
