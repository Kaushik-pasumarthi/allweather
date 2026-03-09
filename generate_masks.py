# import cv2
# import numpy as np
# import os
# import glob
#
#
# def generate_smart_masks(clean_images_dir, output_mask_dir):
#     os.makedirs(output_mask_dir, exist_ok=True)
#     image_paths = glob.glob(os.path.join(clean_images_dir, "*.*"))
#
#     for img_path in image_paths:
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # 1. Blur heavily to destroy fine textures (grass, road grain, leaves)
#         blurred = cv2.GaussianBlur(gray, (15, 15), 0)
#
#         # 2. Find only the strongest edges
#         edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
#
#         # 3. Dilate the edges to connect broken lines of the same object
#         kernel = np.ones((9, 9), np.uint8)
#         dilated = cv2.dilate(edges, kernel, iterations=3)
#
#         # 4. Find the grouped contours
#         contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         mask = np.zeros_like(gray)
#         height, width = gray.shape
#         image_area = height * width
#
#         if contours:
#             for cnt in contours:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 area = w * h
#
#                 # 5. Filter out tiny noise and giant background boxes
#                 # Only keep objects that take up between 1% and 80% of the image
#                 if area > (image_area * 0.01) and area < (image_area * 0.80):
#                     cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=-1)
#
#         filename = os.path.basename(img_path)
#         cv2.imwrite(os.path.join(output_mask_dir, filename), mask)
#         print(f"Generated smart mask for {filename}")
#
#
# # --- RUN THE SCRIPT ---
# clean_dir = "allweather/gt"
# mask_dir = "allweather/masks"
#
# generate_smart_masks(clean_dir, mask_dir)
#


import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO


# def generate_ai_masks(clean_images_dir, output_mask_dir):
#     # 1. Create the output directory if it doesn't exist
#     os.makedirs(output_mask_dir, exist_ok=True)
#
#     # 2. Load the YOLOv8 Nano model (auto-downloads the first time)
#     print("Loading YOLO model...")
#     model = YOLO("yolov8n.pt")
#
#     # 3. Get all the clean ground truth images
#     image_paths = glob.glob(os.path.join(clean_images_dir, "*.*"))
#     print(f"Found {len(image_paths)} images to process.")
#
#     # COCO Dataset Class IDs for street objects:
#     # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
#     target_classes = [0, 1, 2, 3, 5, 7]
#
#     for img_path in image_paths:
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Could not read {img_path}, skipping.")
#             continue
#
#         # 4. Create a pitch-black mask matching the exact image dimensions
#         mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#
#         # 5. Let YOLO find the objects (confidence threshold set to 25%)
#         results = model.predict(source=img, classes=target_classes, conf=0.25, verbose=False)
#
#         # 6. Loop through YOLO's detections and draw the boxes
#         for result in results:
#             if result.boxes is not None:
#                 for box in result.boxes:
#                     # Extract the coordinates of the bounding box
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#                     # Draw a solid white rectangle on the black mask
#                     cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
#
#         # 7. Save the perfect mask
#         filename = os.path.basename(img_path)
#         cv2.imwrite(os.path.join(output_mask_dir, filename), mask)
#         print(f"Generated perfect AI mask for {filename}")
#
#     print("\n✅ All masks generated successfully!")
#
#
# # --- DIRECTORIES ---
# # Make sure these match your exact folder structure!
# clean_dir = "allweather/gt"
# mask_dir = "allweather/masks"
#
# # --- RUN ---
# generate_ai_masks(clean_dir, mask_dir)

##AFTER YOLO IM TRYING OUT THE PREV VERSION CUZ YOLO SUCKS 9/3/26

import cv2
import numpy as np
import os
import glob

# def generate_smart_masks(clean_images_dir, output_mask_dir):
#     os.makedirs(output_mask_dir, exist_ok=True)
#     image_paths = glob.glob(os.path.join(clean_images_dir, "*.*"))
#
#     for img_path in image_paths:
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # 1. Blur heavily to destroy fine textures (grass, road grain, leaves)
#         blurred = cv2.GaussianBlur(gray, (15, 15), 0)
#
#         # 2. Find only the strongest edges
#         edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
#
#         # 3. Dilate the edges to connect broken lines of the same object
#         kernel = np.ones((9, 9), np.uint8)
#         dilated = cv2.dilate(edges, kernel, iterations=3)
#
#         # 4. Find the grouped contours
#         contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         mask = np.zeros_like(gray)
#         height, width = gray.shape
#         image_area = height * width
#
#         if contours:
#             for cnt in contours:
#                 # Calculate the exact area of the contour
#                 area = cv2.contourArea(cnt)
#
#                 # 5. Filter out tiny noise and giant background boxes
#                 # Only keep objects that take up between 1% and 80% of the image
#                 if area > (image_area * 0.01) and area < (image_area * 0.80):
#                     # Draw and fill the actual silhouette
#                     cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
#
#         filename = os.path.basename(img_path)
#         cv2.imwrite(os.path.join(output_mask_dir, filename), mask)
#         print(f"Generated smart mask for {filename}")
#
#
# # --- RUN THE SCRIPT ---
# clean_dir = "allweather/gt"
# mask_dir = "allweather/masks"
#
# generate_smart_masks(clean_dir, mask_dir)


##TRYING OUT THE PREV METHOD WITH CHANGES 9/3/26
import cv2
import numpy as np
import os
import glob

def generate_smart_masks(clean_images_dir, output_mask_dir):
    os.makedirs(output_mask_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(clean_images_dir, "*.*"))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Blur heavily to destroy fine textures (grass, road grain, leaves)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 2. Find only the strongest edges
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # 3. Dilate the edges to connect broken lines of the same object
        kernel = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        # 4. Find the grouped contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        height, width = gray.shape
        image_area = height * width

        if contours:
            for cnt in contours:
                # Calculate the exact area of the contour
                area = cv2.contourArea(cnt)

                # 5. Filter out tiny noise and giant background boxes
                # Only keep objects that take up between 1% and 80% of the image
                if area > (image_area * 0.01) and area < (image_area * 0.80):
                    # Draw and fill the actual silhouette
                    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_mask_dir, filename), mask)
        print(f"Generated smart mask for {filename}")


# --- RUN THE SCRIPT ---
clean_dir = "allweather/gt"
mask_dir = "allweather/masks"

generate_smart_masks(clean_dir, mask_dir)