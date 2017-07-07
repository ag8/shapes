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
Current version: `simpletrain4/train.py`

Current development is undergoing in `simpletrain4/test.py`.  
We're still getting an error: `ValueError: All shapes must be fully defined: [TensorShape([Dimension(None), Dimension(None), Dimension(None)]), TensorShape([])]`
    
See `simpletrain4/qt2.py` for a really simple example of how we're implementing the correct/wrong exampl generation. (Because that's the thing that seems to be broken in the main program right now)

Some of the code based on Tensorflow CIFAR-10 example



### Running instructions

* Tensorflow must be installed.
* Install the following dependencies:
    * tensorflow
    * `sudo apt-get install expect-dev python-numpy`
    * `pip install six pillow requests termcolor keras colorama`