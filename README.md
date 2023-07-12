# DIABETIC RETINOPATHY DETECTION WITH DEEP NEURAL NETWORK

The code is implemented in Jupyter Notebook.
- TensorFlow 2.10 is utilized. 
- The numpy version used is 1.21.6. 
- The pandas version used is 0.24.2.

-Data set:
Due to storage limitations, we were unable to attach the entire dataset.
Normally, there are more than 3000 image, but we could only sent 500.
Each class consist 100 image

-Saving Files:
There are two saved files. The one saved on our PC is named "retina_weights.hdf5," which achieved an accuracy of nearly 80%.
The other saved file, "weights.hdf5," is used during training. However, since we only sent 500 images, our accuracy will not be high.


This project involves a dataset of high-resolution retina images. The images are labeled on a scale of 0 to 4, indicating severity of diabetic retinopathy. 'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'

dataset [link](https://github.com/elif-t/detecting-diabetic-retinopathy/assets/62542563/38bb8eae-97a1-4633-bac4-2d367d99eb2b)

Number of images in Mild = 370 

Number of images in Moderate = 999 

Number of images in No_DR = 1805 

Number of images in Proliferate_DR = 295 

Number of images in Severe = 193

## Data Visualization and Labelling

Visualize 5 images for each class in the dataset

![data](https://github.com/elif-t/detecting-diabetic-retinopathy/blob/main/data_visualize.png)

## Building Res-Block for the Model

In our project, the input shape is (256, 256, 3), which corresponds to an image with a width and height of 256 pixels and 3 color channels (RGB).

Therefore, the number of neurons in the input layer is equal to the total number of input features, which is calculated by multiplying the width, height, and number of channels together:

256 * 256 * 3 = 196,608 neurons in the input layer.

In the model, we used 3 Res-block(3 convolutional and 6 identity block).

The output has only 5 neurons which corresponds to 'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'. 


![blocks](https://github.com/elif-t/detecting-diabetic-retinopathy/blob/main/blocks.png)

![res-net2](https://github.com/elif-t/detecting-diabetic-retinopathy/blob/main/res-net2.png)

![res-net](https://github.com/elif-t/detecting-diabetic-retinopathy/blob/main/Building%20Res-Block%20for%20the%20Model.jpeg)

## Results
**classification report: accuracy, recall and f1-scores**

![report](https://github.com/elif-t/detecting-diabetic-retinopathy/blob/main/report.png)

![output](https://github.com/elif-t/detecting-diabetic-retinopathy/blob/main/output.png)



