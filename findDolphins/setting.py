import tensorflow as tf
# make sure there is a gpu available and set it such that it dynamically
# allocates memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = tf.keras.models.load_model('findDolphins/Ml-test/image_to_number_model.hdf5')
