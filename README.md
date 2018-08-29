# constrained_gm_v2
Second generation constrained generative model code. Better organization of codebase and models.

This code is used to run experiments on constrained generative models, as described in your project with John Cunningham. It requires, in addition
to packages already included in the anaconda virtual environment by default, imageio, joblib, moviepy, tensorflow-gpu, prettytensor, scikit-image, and the 
python scientific computing stack. 

In order to start a new model, write a new config file in cgm_models, and run make_config.py. This will create a model folder with the appropriate
specifications, and you can build a model in this folder. Likewise, you can copy relevant data into a data folder in the model folder directly.
