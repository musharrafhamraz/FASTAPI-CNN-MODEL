import numpy as np
import tensorflow as tf

# Load TensorFlow Lite model
model_path = "model.tflite"  # Path to your TensorFlow Lite model file
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Perform inference
input_data = np.array([[1.5, 2.5]], dtype=np.float32)  # Example input data
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Output:", output_data)