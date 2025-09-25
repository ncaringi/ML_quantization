import tensorflow as tf
import tensorflow_hub as hub
# we use this file to make the int 8 quantization
# load the TensorFlow model from TensorFlow Hub
detector = hub.load("https://kaggle.com/models/tensorflow/centernet-resnet/frameworks/TensorFlow2/variations/50v2-512x512/versions/1")

# Save the TensorFlow model as Savedmodel
saved_model_dir = 'saved_model'
tf.saved_model.save(detector, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the TFLite quantize model
output_path = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\centernet_int8.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_quant_model)
print(f"Modèle TFLite quantifié en int8 sauvegardé avec succès: {output_path}")