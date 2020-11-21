# Asciivate - real-time Structural ASCII-art

Asciivate is a small CUDA app that transforms a video stream into mosaics made of
ASCII characters.

<img src="examples/clocks.gif" width="40%"></img>
<img src="examples/swirls.gif" width="40%"></img>


By default, it converts an available camera stream and displays back the asciivated
version. Additionaly, it can be called with *src* and *dst* arguments to 'asciivate'
a video file.

It works by running rectangular slices of the input image (after some processing)
through a pretrained neural classifier. The best-fitting class represents the index
of the ASCII tile to be put in the original slices' place.

