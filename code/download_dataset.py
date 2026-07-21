import os
import urllib.request
import zipfile

url_images = "http://images.cocodataset.org/zips/val2017.zip"
url_annotations = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

os.makedirs("../dataset", exist_ok=True)

print("Downloading images...")
urllib.request.urlretrieve(url_images, "../dataset/val2017.zip")

print("Downloading annotations...")
urllib.request.urlretrieve(url_annotations, "../dataset/annotations.zip")

print("Extracting images...")
with zipfile.ZipFile("../dataset/val2017.zip", 'r') as zip_ref:
    zip_ref.extractall("../dataset")

print("Extracting annotations...")
with zipfile.ZipFile("../dataset/annotations.zip", 'r') as zip_ref:
    zip_ref.extractall("../dataset")

print("Download complete!")