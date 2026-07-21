import torch
import torchvision
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision import transforms

DATASET_PATH = Path("dataset/val2017")

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

transform = transforms.ToTensor()



def get_predictions(images_df):

    predictions = []

    MAX_IMAGES = 30

    for row in images_df.head(MAX_IMAGES).itertuples():

        image_path = DATASET_PATH / row.file_name

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)

        with torch.no_grad():
            preds = model([image_tensor])[0]

        boxes = preds["boxes"]
        labels = preds["labels"]
        scores = preds["scores"]

        for i in range(len(boxes)):

            predictions.append({
                "pred_id": len(predictions),
                "image_id": row.id,
                "label_id": int(labels[i]),
                "xmin": float(boxes[i][0]),
                "ymin": float(boxes[i][1]),
                "xmax": float(boxes[i][2]),
                "ymax": float(boxes[i][3]),
                "score": float(scores[i])
            })

    preds_df = pd.DataFrame(predictions)

    return preds_df