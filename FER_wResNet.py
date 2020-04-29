import numpy as np
import pandas as pd
import keras
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras import backend as K
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense 
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback
import h5py


# Folder where logs and models are stored
folder = 'logs/ResNet-50'
# Size of the images
img_height, img_width = 197, 197
# Parameters
num_classes         = 7     # ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
epochs_top_layers   = 5
epochs_all_layers   = 100
batch_size          = 128
train_dataset	= 'C:/Users/Hilal/.spyder-py3/olduuuu/Datasets/Train/train.csv'
eval_dataset 	= 'C:/Users/Hilal/.spyder-py3/olduuuu/Datasets/Test/test.csv'
base_model = VGGFace(
    model       = 'resnet50',
    include_top = False,
    weights     = 'vggface',
    input_shape = (img_height, img_width, 3))
# Places x as the output of the pre-trained model
x = base_model.output

# Flattens the input. Does not affect the batch size
x = Flatten()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
model.summary()

def preprocess_input(x):
    x -= 128.8006 # np.mean(train_dataset)
    return x
def get_data(dataset):
    file_stream = file_io.FileIO(dataset, mode='r')
    data = pd.read_csv(file_stream)
    pixels = data['pixels'].tolist()
    images = np.empty((len(data), img_height, img_width, 3))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
        single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 3, mode = 'constant') # Dimension: 139x139x3 (Bicubic)
        ret = np.empty((img_height, img_width, 3))  
        ret[:, :, 0] = single_image
        ret[:, :, 1] = single_image
        ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1
    
    images = preprocess_input(images)
    labels = to_categorical(data['emotion'])

    return images, labels    

# Data preparation
train_data_x, train_data_y  = get_data(train_dataset)
val_data  = get_data(eval_dataset)
train_datagen = ImageDataGenerator(
    rotation_range  = 10,
    shear_range     = 10, # 10 degrees
    zoom_range      = 0.1,
    fill_mode       = 'reflect',
    horizontal_flip = True)
train_generator = train_datagen.flow(train_data_x,train_data_y, batch_size  = batch_size)
for layer in base_model.layers:
    layer.trainable = False
model.compile(
    optimizer   = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0), 
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])
tensorboard_top_layers = TensorBoard(
	log_dir         = folder + '/logs_top_layers',
	histogram_freq  = 0,
	write_graph     = True,
	write_grads     = False,
	write_images    = True)
model.fit_generator(
    generator           = train_generator,
    steps_per_epoch     = len(train_data_x) // batch_size,  # samples_per_epoch / batch_size
    epochs              = epochs_top_layers,                            
    validation_data     = val_data,
    callbacks           = [tensorboard_top_layers])

# Fine-tuning of all the layers
for layer in model.layers:
    layer.trainable = True
model.compile(
    optimizer   = keras.optimizers.SGD(lr = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])

# This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, 
tensorboard_all_layers = TensorBoard(
    log_dir         = folder + '/logs_all_layers',
    histogram_freq  = 0,
    write_graph     = True,
    write_grads     = False,
    write_images    = True)

def scheduler(epoch):
    updated_lr = K.get_value(model.optimizer.lr) * 0.5
    if (epoch % 3 == 0) and (epoch != 0):
        K.set_value(model.optimizer.lr, updated_lr)
        print(K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

# Learning rate scheduler
    # schedule: a function that takes an epoch index as input (integer, indexed from 0) and current learning
    #           rate and returns a new learning rate as output (float)
reduce_lr = LearningRateScheduler(scheduler)


# Reduce learning rate when a metric has stopped improving
	# monitor: 	Quantity to be monitored
	# factor: 	Factor by which the learning rate will be reduced. new_lr = lr * factor
	# patience:	Number of epochs with no improvement after which learning rate will be reduced
	# mode: 	One of {auto, min, max}
	# min_lr:	Lower bound on the learning rate
reduce_lr_plateau = ReduceLROnPlateau(
	monitor 	= 'val_loss',
	factor		= 0.5,
	patience	= 3,
	mode 		= 'auto',
	min_lr		= 1e-8)

# Stop training when a monitored quantity has stopped improving
	# monitor:		Quantity to be monitored
	# patience:		Number of epochs with no improvement after which training will be stopped
	# mode: 		One of {auto, min, max}
early_stop = EarlyStopping(
	monitor 	= 'val_loss',
	patience 	= 10,
	mode 		= 'auto')

# Save the model after every epoch
	# filepath:       String, path to save the model file
	# monitor:        Quantity to monitor {val_loss, val_acc}
	# save_best_only: If save_best_only=True, the latest best model according to the quantity monitored will not be overwritten
	# mode:           One of {auto, min, max}. If save_best_only = True, the decision to overwrite the current save file is made based on either
	#			      the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should
	#			      be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity
	# period:         Interval (number of epochs) between checkpoints
check_point = ModelCheckpoint(
	filepath		= 'ResNet-50_{epoch:02d}_{val_loss:.2f}.h5',
	folder 			= folder,
	monitor 		= 'val_loss', # Accuracy is not always a good indicator because of its yes or no nature
	save_best_only	= True,
	mode 			= 'auto',
	period			= 1)

# We train our model again (this time fine-tuning all the resnet blocks)
model.fit_generator(
    generator           = train_generator,
    steps_per_epoch     = len(train_data_x) // batch_size,  # samples_per_epoch / batch_size 
    epochs              = epochs_all_layers,                        
    validation_data     = val_data,
    callbacks           = [tensorboard_all_layers, reduce_lr, reduce_lr_plateau, early_stop, check_point])

# SAVING ##############################################################################################################################################

# Saving the model in the workspace
model.save(folder + '/ResNet-50.h5')
# Save model.h5 on to google storage
with file_io.FileIO('ResNet-50.h5', mode='r') as input_f:
    with file_io.FileIO(folder + '/ResNet-50.h5', mode='w+') as output_f:  # w+ : writing and reading
        output_f.write(input_f.read())
