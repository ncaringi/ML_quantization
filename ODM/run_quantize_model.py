import numpy as np
import tensorflow as tf
import cv2
import os
import json
import time
# We use this file to run the quantize model and extract the json result file
# Paths to TFLite models
model_paths = {
    'float32': r'C:\Users\Admin\Documents\TU Wien\projets\centernet\models\centernet_float32.tflite',
    'float16': r'C:\Users\Admin\Documents\TU Wien\projets\centernet\models\centernet_float16.tflite',
    'int8': r'C:\Users\Admin\Documents\TU Wien\projets\centernet\models\centernet_int8.tflite'
}

# File with images
image_dir = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_petit'
# output folder for the results
output_dir = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_tested'

# load and resize the images with the correct dimensions for the model.
def load_and_preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]
    #print(f"Original dimensions: {orig_width}x{orig_height}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image

# INference Function for a given TFlite model
def run_inference(interpreter, input_data, image_id):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_type = input_details[0]['dtype']

    # Display Input details
    #print("Input details:", input_details)

    # Scaling for Int8 
    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale) + input_zero_point
        input_data = np.around(input_data).astype(np.int8)
    else:
        input_data = input_data.astype(input_type)

    input_data = np.expand_dims(input_data, axis=0)

    # Put the input tensor size to (1, 512, 512, 3)
    interpreter.resize_tensor_input(input_details[0]['index'], [1, 512, 512, 3])
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # display the input and output tensor values
    #print("Input tensor values:", interpreter.get_tensor(input_details[0]['index']))
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print("Output tensor values:", output_data)

    # recalibrate for the int8 model
    output_type = output_details[0]['dtype']
    if output_type == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    # Extract the result and put them in the right format for coco eval
    num_detections = int(output_data[0])
    detection_boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    detection_classes = interpreter.get_tensor(output_details[2]['index'])[0].astype(int)
    detection_scores = interpreter.get_tensor(output_details[3]['index'])[0]

    results = []
    for i in range(num_detections):
        if detection_scores[i] >= 0.5:  # threshold 
            box = detection_boxes[i]
            ymin, xmin, ymax, xmax = box
            width, height = xmax - xmin, ymax - ymin
            bbox = [float(xmin), float(ymin), float(width), float(height)]
            result = {
                "image_id": int(image_id),
                "category_id": int(detection_classes[i]),
                "bbox": bbox,
                "score": float(detection_scores[i])
            }
            results.append(result)

    print(f"Results for image {image_id}: {results}")
    return results

# load the models and create output folders
interpreters = {}
for quant_type, model_path in model_paths.items():
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    interpreters[quant_type] = interpreter
    
    # get input details for each model
    input_details = interpreter.get_input_details()
    #print(f"{quant_type} model input details: {input_details}")
    
    # Create a sub-file for each quantization
    os.makedirs(os.path.join(output_dir, quant_type), exist_ok=True)

inference_time_results = []
# Iterate over each image and make the inference
for quant_type, interpreter in interpreters.items():
    start_time = time.time()
    all_results = []
    counter = 0
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = load_and_preprocess_image(image_path)
        image_id = int(os.path.splitext(image_file)[0])  

        results = run_inference(interpreter, image, image_id)
        all_results.extend(results)
        counter += 1
        print(f"{quant_type} : image n°{counter}")
    print(f'Nombre total image pour {quant_type} : {counter}')
    
    # Check the results before saving them
    print(f"Exemple de résultats pour {quant_type} : {all_results[:5]}")

    # Save the results in a JSON file with the COCO format for each model
    output_file = os.path.join(output_dir, quant_type, "coco_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Inference results saved to {output_file} for {quant_type} model")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time pour {quant_type} : {total_time}')
    inference_time = total_time / counter
    print(f'Inference time du modèle {quant_type} : {inference_time}')
    inference_time_results.append(inference_time)

print(f'Resultats finaux : {inference_time_results}')