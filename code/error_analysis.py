import pandas as pd
import torch
from torchvision.ops import box_iou

PREDS_DF_COLUMNS = [
    "pred_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
]

TARGETS_DF_COLUMNS = [
    "target_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
]

def analyze_errors(targets_df, preds_df, iou_threshold=0.5):

    errors = []

    for image_id in targets_df["image_id"].unique():

        gt = targets_df[targets_df["image_id"] == image_id]
        pr = preds_df[preds_df["image_id"] == image_id]

        if len(gt) == 0 or len(pr) == 0:
            continue

        gt_boxes = torch.tensor(gt[["xmin","ymin","xmax","ymax"]].values)
        pr_boxes = torch.tensor(pr[["xmin","ymin","xmax","ymax"]].values)

        ious = box_iou(pr_boxes, gt_boxes)

        for i, pred_row in enumerate(pr.itertuples()):

            best_iou, gt_idx = ious[i].max(0)

            gt_idx = int(gt_idx.item())

            if best_iou >= iou_threshold:

                target = gt.iloc[gt_idx]

                if pred_row.label_id == target.label_id:
                    err = "OK"
                else:
                    err = "CLS"

                errors.append({
                    "pred_id": pred_row.pred_id,
                    "target_id": target.target_id,
                    "error_type": err
                })

            elif best_iou > 0:
                errors.append({
                    "pred_id": pred_row.pred_id,
                    "target_id": None,
                    "error_type": "LOC"
                })

            else:
                errors.append({
                    "pred_id": pred_row.pred_id,
                    "target_id": None,
                    "error_type": "BKG"
                })

    errors_df = pd.DataFrame(errors)

    matched_targets = set(errors_df["target_id"].dropna())
    all_targets = set(targets_df["target_id"])

    missing_targets = all_targets - matched_targets

    for t in missing_targets:
        errors.append({
            "pred_id": None,
            "target_id": t,
            "error_type": "MISS"
        })

    errors_df = pd.DataFrame(errors)

    return errors_df