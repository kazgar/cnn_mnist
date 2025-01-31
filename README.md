I have created a a CNN that closely resembles the TinyVGG architecture. After training, it achieves 98.7% accuracy on test set and 99.02% accuracy on training set. 

In order to use it, install all the dependencies listed in requirements.txt. 

Steps: 
<li> Create, train and save the model using model.ipynb notebook.
<li> Download a picture of handwritten digit (yours or from internet - up to you).
<li> Run the main.py script by executing "python@ model.py (saved_model_path) (image_path)" where @ is the python version you are using. 

PS From what I know PyTorch works best with python version up to and including 3.12, some dependencies are not supported for 3.13 afaik.

Have fun!
