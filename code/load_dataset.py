import json
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("dataset")


def load_dataset():

    annotations_path = DATASET_PATH / "annotations/instances_val2017.json"

    with open(annotations_path) as f:
        data = json.load(f)

    images_df = pd.DataFrame(data["images"])
    targets_df = pd.DataFrame(data["annotations"])

    targets_df["target_id"] = targets_df.index

    targets_df["xmin"] = targets_df["bbox"].apply(lambda x: x[0])
    targets_df["ymin"] = targets_df["bbox"].apply(lambda x: x[1])
    targets_df["xmax"] = targets_df["bbox"].apply(lambda x: x[0] + x[2])
    targets_df["ymax"] = targets_df["bbox"].apply(lambda x: x[1] + x[3])

    targets_df = targets_df.rename(columns={"category_id": "label_id"})

    print("Images:", images_df.shape)
    print("Annotations:", targets_df.shape)

    return images_df, targets_df