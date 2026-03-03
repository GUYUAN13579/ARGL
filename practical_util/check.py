import torch  
import numpy
import tensorflow as tf  
print(torch.__version__)
print(torch.version.cuda)  
print(numpy.__version__)
gpus = tf.config.list_physical_devices('GPU')  
if gpus:  
    print("Available GPUs:")  
    for gpu in gpus:  
        print(gpu)  
else:  
    print("No GPU available")  