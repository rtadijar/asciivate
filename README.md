# Asciivate

![](images/asciivate.gif)

Asciivate is a CUDA app that transforms a video stream into structure-based ASCII art. It uses a pretrained neural network to classify image patches into the best-fitting character tile. It is capable of real-time (24fps+) performance on HD resolution.<sup>1</sup>

Run without arguments, it converts the default camera stream and displays it back to the user.
Additionaly, it can be called with *src* and *dst* arguments to convert a video file.

The repository also contains code for creating a labeled dataset and training classifiers. The label of a single image patch is decided by comparing it with all the character tiles used. The most "similar" tile wins. The similarity metric is based on SSIM, with several modifications that make the result more aesthetically pleasing.

###### <sup>1</sup>Mileage may vary depending on one's GPU. The use of cv::imshow() may underrepresent the actual framerate achievable by the machine.

## Examples

<img src="images/clocks.gif" width="45%"></img>
<img src="images/swirls.gif" width="45%"></img>

## Repository structure

As of now, training models to be used by the app is not a streamlined process. The repo contains code that evolved as it was used during research/development of the application, and the focus was on getting things to work. When time permits, I'll expand on the code so more of it can be immediately useful.

* src/ 

    Contains the Asciivate source code. `src/dataset` contains code for creating a labeled dataset.
* notebooks/

    Various notebooks used in the process of developping the application. Notably, `asciivate.ipyb` contains code that trained the classifier.
* res/

    Contains a trained model and tileset to be used by the application at runtime.
* images/

    Various images used in this readme.

## Requirements

* NVIDIA GPU + CUDA
* OpenCV (CUDA-enabled)

## Installation

1. Run `cmake` in a build directory
2. Run `make`
3. Make sure `model.dat` and `texture.png` are in the executable directory.
4. Done!

## Acknowledgments

My thanks goes to the creators of [these](http://www.cse.cuhk.edu.hk/~ttwong/papers/asciiart/asciiart.pdf) [papers](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12597) and others which served to clarify what it is I wanted to do with this project.
