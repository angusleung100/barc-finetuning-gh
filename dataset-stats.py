from datasets import load_dataset
import pandas as pd

dataset = load_dataset("imagefolder", data_dir="g:/My Drive/AI Consistent Generation/train_test", split="train") # https://huggingface.co/docs/datasets/en/image_dataset

print(dataset)

dataset = dataset.to_pandas()   # Convert to pandas

print(dataset)

print("Number Images Per Label")
print("=======================")

id2label = {0: "Realistic_Bad_Anatomy", 1: "Realistic_Good_Anatomy", 2: "Unrealistic_Bad_Anatomy", 3: "Unrealistic_Good_Anatomy"}

label_counts = dataset['label'].value_counts()

for label_id in id2label:

    total_images = len(dataset)
    total_images_for_label = label_counts.get(label_id)

    print(id2label[label_id].replace("_", " ")+":", total_images_for_label, "("+str(round((total_images_for_label/total_images)*100, 2))+"%)")

print("\n")

print("Total Number of Images: ", len(dataset))