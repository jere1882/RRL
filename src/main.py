import common
import model_selection
import preprocessing
import dimensionality_reduction
import inspection 
import imbalance
import covariate_shift

model_selection.init("/home/jere/carpyncho/","/home/jere/Desktop/ms/")
preprocessing.init("/home/jere/carpyncho/","/home/jere/Desktop/preprocessing/")
dimensionality_reduction.init("/home/jere/carpyncho/","/home/jere/Desktop/feature-selection/","/home/jere/Desktop/inspection/")
inspection.init("/home/jere/carpyncho/","/home/jere/Desktop/inspection/")
imbalance.init("/home/jere/carpyncho/","/home/jere/Desktop/imbalance/")
covariate_shift.init("/home/jere/carpyncho/","/home/jere/Desktop/covariate-shift/")