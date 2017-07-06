# shapes

## Shape generation
Run `shape_generation/main.py`.

Edit flags in `shape_generation/nshapegenflags.py`:
* `IMAGE_NUM` - the number of images to be generated
* `DIM` - dimension of images
* `COLOR` - what color to use for the shapes
* `RANDOM_COLOR` - boolean; whether to use random colors for the shapes instead. Overrides `COLOR`. (This is nice for quickly being able to tell whether two shapes match or not)
* `ROTATE` - Whether to rotate the generated shape pieces
* `SIMPLE` - **(experimental!)** boolean; whether to use simple geometric shapes or more complex geometric shapes
* `CUT` - **(not fully implemented)** which cut style to use when cutting the shapes


## Shape training
Current version: `simpletrain3.py`
    (Seems to work, but I haven't beel able to test it on Titan yet)  
Working on `simpletrain4`, which will merge several queues and prevent unnecessary errors

Some of the code based on Tensorflow CIFAR-10 example



### Running instructions

* Tensorflow must be installed.
* Install the following dependencies:
    * `pip install termcolor colorama`
    * `sudo apt-get install expect-dev`