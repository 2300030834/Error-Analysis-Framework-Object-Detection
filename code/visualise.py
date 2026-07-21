import matplotlib
matplotlib.use('TkAgg')
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_boxes(image_path, preds_df, targets_df, image_id):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found:", image_path)
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------------------------
    # 🟢 Ground Truth (GREEN)
    # ---------------------------
    gt = targets_df[targets_df["image_id"] == image_id]

    count = 0
    for _, row in gt.iterrows():

        # small boxes remove
        if (row["xmax"] - row["xmin"] < 40) or (row["ymax"] - row["ymin"] < 40):
            continue

        if count >= 8:   # limit boxes
            break

        count += 1

        cv2.rectangle(image,
                      (int(row["xmin"]), int(row["ymin"])),
                      (int(row["xmax"]), int(row["ymax"])),
                      (0,255,0), 2)

    # ---------------------------
    # 🔴 Predictions (RED)
    # ---------------------------
    preds = preds_df[preds_df["image_id"] == image_id]

    # score filter
    preds = preds[preds["score"] > 0.7]

    boxes = []
    scores = []

    for _, row in preds.iterrows():
        x1 = int(row["xmin"])
        y1 = int(row["ymin"])
        x2 = int(row["xmax"])
        y2 = int(row["ymax"])

        boxes.append([x1, y1, x2-x1, y2-y1])
        scores.append(float(row["score"]))

    # 🔥 Non-Max Suppression (remove overlaps)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.7, nms_threshold=0.4)

    if len(indices) > 0:
        for i in indices.flatten():

            x, y, w, h = boxes[i]

            cv2.rectangle(image,
                          (x, y),
                          (x+w, y+h),
                          (255,0,0), 2)

    # ---------------------------
    # Display
    # ---------------------------
    plt.imshow(image)
    plt.title("Green = Ground Truth, Red = Prediction (Clean)")
    plt.axis("off")
    plt.show()