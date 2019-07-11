# Jetson Nano
## Pose Estimation with Jetson Nano

![open_pose_jetson_nano](/images_jetson/openpose.gif)

### Hardware
- Jetson nano
- Raspi Cam V2 or USB Web Cam

### Setup
Execute following commands for installing TensorFlow and tf-pose-estimation:

```sh
$ git clone https://github.com/karaage0703/jetson-nano-tools
$ cd jetson-nano-tools
$ ./install-tensorflow.sh
$ ./install-open-pose.sh
```
### How to use
With Raspi Cam V2

```
$ cd ~/tf-pose-estimation
$ python3 run_jetson_nano.py --model=mobilenet_v2_small --resize=320x176
```

With USB Web Cam

```sh
$ cd ~/tf-pose-estimation
$ python3 run_webcam.py --model=mobilenet_v2_small --resize=320x176 --camera=1
```

## Skeleton Sequencer with Jetson Nano

![skeleton_sequencer_02](/images_jetson/ss_02.jpg)

![skeleton_sequencer_01](/images_jetson/ss_01.jpg)

### Hardware
- Jetson nano
- Raspi Cam V2
- Pocket Miku (as MIDI Device)

### Setup
For setup Sekeleton Sequencer needs 2 steps.

1. Kernel build
1. Install pygame

#### Kernel Build for MIDI
Execute following commands for kernel build for MIDI:

```sh
$ cd && mkdir kernel && cd kernel
$ wget https://developer.nvidia.com/embedded/dlc/l4t-sources-32-1-jetson-nano -O l4t-sources-32-1-jetson-nano.tar.gz
$ tar xvf l4t-sources-32-1-jetson-nano.tar.gz
$ cd public_sources
$ tar xvf kernel_src.tbz2
$ cd kernel/kernel-4.9
$ wget -O .config https://raw.githubusercontent.com/karaage0703/jetson-nano-tools/master/kernel_config/config_midi
$ make oldconfig
$ make prepare
$ make modules_prepare
$ make -j4 Image && make -j4 modules
$ sudo make modules_install
```
#### Install pygame
Execute following commands for installing pygame:

```sh
$ sudo apt-get update
$ sudo apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev
$ sudo apt-get install libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev
$ sudo apt-get install libfreetype6-dev
$ sudo apt-get install libportmidi-dev
$ sudo apt-get install python3-pip
$ pip3 install pygame
```



### How to use
Execute following commands:

```sh
$ cd ~/tf-pose-estimation
$ python3 skeleton_sequencer -d=jetson_nano_raspi_cam
```

# tf-pose-estimation

'Openpose', human pose estimation algorithm, have been implemented using Tensorflow. It also provides several variants that have some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**

**You can even run this on your macbook with a descent FPS!**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

| CMU's Original Model</br> on Macbook Pro 15" | Mobilenet-thin </br>on Macbook Pro 15" | Mobilenet-thin</br>on Jetson TX2 |
|:---------|:--------------------|:----------------|
| ![cmu-model](/etcs/openpose_macbook_cmu.gif)     | ![mb-model-macbook](/etcs/openpose_macbook_mobilenet3.gif) | ![mb-model-tx2](/etcs/openpose_tx2_mobilenet3.gif) |
| **~0.6 FPS** | **~4.2 FPS** @ 368x368 | **~10 FPS** @ 368x368 |
| 2.8GHz Quad-core i7 | 2.8GHz Quad-core i7 | Jetson TX2 Embedded Board | 

Implemented features are listed here : [features](./etcs/feature.md)

## Important Updates

- 2019.3.12 Add new models using mobilenet-v2 architecture. See : [experiments.md](./etcs/experiments.md)
- 2018.5.21 Post-processing part is implemented in c++. It is required compiling the part. See: https://github.com/ildoonet/tf-pose-estimation/tree/master/src/pafprocess
- 2018.2.7 Arguments in run.py script changed. Support dynamic input size.

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-pose-estimation
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Package Install

Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-openpose
$ python setup.py install  # Or, `pip install -e .`
```

## Models & Performances

See [experiments.md](./etc/experiments.md)

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files. You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

## Demo

### Test Inference

You can test the inference feature with a single image.

```
$ python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

The image flag MUST be relative to the src folder with no "~", i.e:
```
--image ../../Desktop
```

Then you will see the screen as below with pafmap, heatmap, result and etc.

![inferent_result](./etcs/inference_result2.png)

### Realtime Webcam

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
```

Then you will see the realtime webcam screen with estimated poses as below. This [Realtime Result](./etcs/openpose_macbook13_mobilenet2.gif) was recored on macbook pro 13" with 3.1Ghz Dual-Core CPU.

## Python Usage

This pose estimator provides simple python classes that you can use in your applications.

See [run.py](run.py) or [run_webcam.py](run_webcam.py) as references.

```python
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
humans = e.inference(image)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
```

If you installed it as a package,

```python
import tf_pose
coco_style = tf_pose.infer(image_path)
```

## ROS Support

See : [etcs/ros.md](./etcs/ros.md)

## Training

See : [etcs/training.md](./etcs/training.md)

## References

See : [etcs/reference.md](./etcs/reference.md)