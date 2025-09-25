import json
# This code is to make a list of every ID from the images in our dataset because we don't use the whole dataset ( only 1000 images)
# Path to  JSON  result file 
resFile = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_tested\float32\coco_results.json'

# Load the json result file
with open(resFile, 'r') as f:
    results = json.load(f)

# Extract the ID's and delete the duplicate
image_ids = list(set(item['image_id'] for item in results))

# Save the list of ID's in a JSON file
output_file = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_tested\image_ids.json'
with open(output_file, 'w') as f:
    json.dump(image_ids, f)

print(f"List image_id saved in  {output_file}")