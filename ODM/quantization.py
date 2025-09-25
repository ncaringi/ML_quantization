# quantif.py
import tensorflow as tf
import tensorflow_hub as hub
#we use this file to make the float quantization
# Load the TensorFlow model
detector = hub.load("https://kaggle.com/models/tensorflow/centernet-resnet/frameworks/TensorFlow2/variations/50v2-512x512/versions/1")

def convert_and_quantize_model(detector, quantization_type, output_path):
    concrete_func = detector.signatures['serving_default']
    
    # Create the converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], detector)
    
    
    if quantization_type == 'float32':
        # No specific quantization for float32
        pass
    elif quantization_type == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    

    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Modèle sauvegardé avec succès: {output_path}")

# Convert with every config
convert_and_quantize_model(detector, 'float32', 'centernet_float32.tflite')
convert_and_quantize_model(detector, 'float16', 'centernet_float16.tflite')
