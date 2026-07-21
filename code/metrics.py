import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class MyMeanAveragePrecision:

    def __init__(self, iou_threshold):

        self.metric = MeanAveragePrecision(
            iou_thresholds=[iou_threshold]
        )

    def __call__(self, targets_df, preds_df):

        targets = []
        preds = []

        image_ids = set(targets_df["image_id"])

        for img_id in image_ids:

            gt = targets_df[targets_df["image_id"] == img_id]
            pr = preds_df[preds_df["image_id"] == img_id]

            targets.append({
                "boxes": torch.tensor(gt[["xmin","ymin","xmax","ymax"]].values, dtype=torch.float32),
                "labels": torch.tensor(gt["label_id"].values, dtype=torch.int64)
            })

            preds.append({
                "boxes": torch.tensor(pr[["xmin","ymin","xmax","ymax"]].values, dtype=torch.float32),
                "labels": torch.tensor(pr["label_id"].values, dtype=torch.int64),
                "scores": torch.tensor(pr["score"].values, dtype=torch.float32)
            })

        self.metric.update(preds, targets)
        result = self.metric.compute()

        self.metric.reset()

        return result["map"].item()