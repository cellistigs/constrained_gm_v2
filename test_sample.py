## basic testing file:
from input import temppipeline_0,temppipeline_1
# content of test_sample.py
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5

def test_pipeline():
    ## Replace this with a fake database. 
    filenames = ['datadirectory/toydynamics_nograv/Video_ball_color_med_motion_w_b0encoder_train.tfrecords']
