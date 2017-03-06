# Art Generation (Style Transfer)

Given an input content image and corresponding style images, combine them to form a final image that has the semantic details and features of the content image but has the artistic style of the style image(s). This allows us to generate unique images by combining various style images and content images. Response for Sirajology's challenge [here](https://youtu.be/Oex0eWoU7AQ)

## Results

#### Run 1

* Input Content and Style Images

<img src="images/hugo.jpg" alt="" width="30%"> <img src="images/styles/wave.jpg" alt="" width="30%"> <img src="images/styles/forest.jpg" alt="" width="30%">

* Final Output

![Output a1](Outputs/Images/Run1/result9.bmp?raw=true "Output a1")

#### Run 2

* Input Content and Style Images

<img src="images/hugo.jpg" alt="" width="30%"> <img src="images/styles/gothic.jpg" alt="" width="30%"> <img src="images/styles/starry_night.jpg" alt="" width="30%">

* Final Output

![Output b1](Outputs/Images/Run2/result9.bmp?raw=true "Output b1")


## Implementation Details

* Uses VGG16 pretrained model to perform style transfer at various layers in the network
* Performs normalization and converts RGB to BGR as required by VGG16
* Uses multiple style images which are transferred onto the specified content image
* Video style transfer takes a video as the input and then generates an output video by performing style transfer on the individual frames (can use multiple styles as well)
* Frames concatenated to form a batch input which helps speed up the computation (much faster than running the network for each and every frame seperately)


## Dependencies

* Run `pip install -r requirements.txt` to install the necessary dependencies
* Install OpenCV


## Usage

If it doesn't exist, create a file called ~/.keras/keras.json and make sure it looks like the following:

   ````
   {
       "image_dim_ordering": "tf",
       "epsilon": 1e-07,
       "floatx": "float32",
       "backend": "tensorflow"
   }
   ````

You can then run the code by `python styletransfer.py` or the video version by `python styletransfervid.py`


### Credits

Artistic-Style-Transfer [hnarayanan](https://github.com/hnarayanan/artistic-style-transfer) and Sirajology for the starter code




