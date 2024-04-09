import tensorflow as tf

# Load the HDF5 model
h5_model_path = "potatoes.h5"  # Replace with the path to your .h5 model file
h5_model = tf.keras.models.load_model(h5_model_path, compile=False)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
tflite_model_path = "converted_model.tflite"  # Path to save the converted .tflite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("Model converted successfully to TensorFlow Lite format:", tflite_model_path)

