import os.path
import cv2

import gdown
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


class RoiExtractor:
    """
    This class is responsible for extracting regions of interest (ROIs) from images using a YOLO model.
    It identifies specific regions in the image based on the model's predictions and categorizes them into
    different types such as language, frequency, and medication, then extract the text from them.
    """

    def __init__(self):
        """
        Initializes the RoiExtractor class.
        It downloads the YOLO model if it does not exist locally and loads the model.
        """
        self.local_path = './Backend/Resources/weights/Roi_extractor.pt'
        if not os.path.exists(self.local_path):
            self.download_model()
        self.yolo = YOLO(self.local_path)
        self.classes = self.yolo.names

    def extract_info(self, img_path):
        """
        Predicts the regions of interest in the given image using the YOLO model
        and gets the description associated with each one.

        :param img_path: the path to the image
        :return: parts of the image each one include the bounding box, the description of it and the predicted label
        """
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        boxes = self.get_ROI(img)

        img = cv2.imread(img_path)
        out = self.add_lang_type(boxes, img)

        return out

    def get_ROI(self, img):
        """
        This function is used to get the region of interest (ROI) from the image using the YOLO model.

        :param img: the image which should be processed
        :return: bounding boxes, confidence scores and class ids
        """
        # get the region of interest (ROI) from the image
        roi = self.yolo.predict(img, conf=0.5, verbose=False)
        w, h = img.shape[1], img.shape[0]

        result = roi[0].obb
        result = sorted(result, key=lambda x: x.xyxy[0][1].cpu().numpy())  # sort according to the y-axis

        # get the bounding boxes, confidence scores and class ids
        conf = []
        xyxy = []
        cls_ids = []
        for res in result:
            cls_ids.append(int(res.cls[0].cpu().numpy()))
            conf.append(res.conf[0].cpu().numpy())

            x1, y1, x2, y2 = res.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h

            xyxy.append((x1, y1, x2, y2))

        return xyxy, conf, cls_ids

    def reserve_values(self, opt1, opt2, target):
        """
        This function is used to reserve the values of the language and type of the prescription.
        The reserved vlue is determined with the priority where the first parameter has the highest priority,
        the second is less and the last is the least.

        param opt1: the first option
        :param opt2: the second option
        :param target: the target value
        :return: the reserved value
        """
        if opt1 is not None:
            target = opt1
        elif opt2 is not None:
            target = opt2
        return target

    def add_lang_type(self, boxes, img):
        """
        This function is used to add the language and type of the prescription to the bounding boxes.

        :param boxes: the bounding boxes, confidence scores and class ids
        :param img: the image which should be processed
        :return: parts of the image each one include the bounding box, the description of it and the predicted label
        """
        bbox, conf, cls_ids = boxes

        num_med = 0
        num_freq = 0
        num_lang = 0
        num_types = 0

        # count the number of each class
        for i in range(len(cls_ids)):
            if cls_ids[i] == 2:
                num_freq += 1
            elif cls_ids[i] == 3:
                num_med += 1

            if cls_ids[i] <= 1:
                num_lang += 1
            else:
                num_types += 1

        parts = []
        frequency_index = 0
        medicine_index = 0

        remove_list = []

        for i in range(len(bbox)):
            if i in remove_list:
                continue
            best_iou, box_idx, lang_res, type_res = 0, None, None, None
            lang_1, type_1 = self.get_region_info(cls_ids[i], cls_ids[i])
            lang_res, type_res = lang_1, type_1

            for j in range(i + 1, len(bbox)):
                if j in remove_list:
                    continue
                cur_iou = self.iou(bbox[i], bbox[j])
                if cur_iou > 0.35:
                    remove_list.append(j)
                    lang_2, type_2 = self.get_region_info(cls_ids[j], cls_ids[j])

                    if cur_iou > best_iou:  # extract the info when the boxes refer to the same region
                        lang_res = self.reserve_values(lang_1, lang_2, lang_res)
                        type_res = self.reserve_values(type_1, type_2, type_res)
                        best_iou, box_idx = cur_iou, j
                        continue

                    lang_res = self.reserve_values(lang_1, lang_res, lang_2)
                    type_res = self.reserve_values(type_1, type_res, type_2)

            if best_iou != 0:
                lang, prescription_type = self.get_region_info(cls_ids[i], cls_ids[box_idx])

                lang = lang if lang != None else lang_res
                prescription_type = prescription_type if prescription_type != None else type_res

                # crop the region of interest
                x1, y1, x2, y2 = bbox[i]
                x1_1, y1_1, x2_1, y2_1 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(
                    y2 * img.shape[0])

                x1, y1, x2, y2 = bbox[box_idx]
                x1_2, y1_2, x2_2, y2_2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(
                    y2 * img.shape[0])

                x1 = min(x1_1, x1_2)
                y1 = min(y1_1, y1_2)
                x2 = max(x2_1, x2_2)
                y2 = max(y2_1, y2_2)
                box = (x1, y1, x2, y2)

                # get the label of the ROI
                if prescription_type == 3:  # 3 -> medicine
                    medicine_index += 1
                elif prescription_type == 2:  # 2 -> frequency
                    frequency_index += 1

                # add the label with the other info into the list
                parts.append((box, (prescription_type, lang)))
                # break

            else:
                lang, prescription_type = self.get_region_info(cls_ids[i], cls_ids[i])
                lang = lang if lang != None else lang_res
                prescription_type = prescription_type if prescription_type != None else type_res

                if lang is not None:
                    # crop the region of interest
                    x1, y1, x2, y2 = bbox[i]
                    x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(
                        y2 * img.shape[0])
                    box = (x1, y1, x2, y2)

                    # get the label of the ROI
                    if prescription_type == 3:  # 3 -> medicine
                        medicine_index += 1
                    elif prescription_type == 2:  # 2 -> frequency
                        frequency_index += 1

                    parts.append((box, (prescription_type, lang)))

            if frequency_index == num_freq and medicine_index == num_med:
                break

        i = 0

        while i < len(parts) - 1:
            coord1, (type1, lang1) = parts[i]
            coord2, (type2, lang2) = parts[i + 1]

            x1, y1, x2, y2 = coord1
            x3, y3, x4, y4 = coord2

            same_type = type1 == type2
            same_line = abs(y1 - y3) < 50 or abs(y2 - y4) < 50
            not_overlapping = not (x1 < (x3 + x4) / 2 < x2) and not (x3 < (x1 + x2) / 2 < x4)

            # Optional: horizontal distance check to avoid merging far items
            horiz_gap = min(abs(x3 - x2), abs(x1 - x4))

            # Check if both parts are of the same prescription type, on the same line and not overlapping
            if same_type and same_line and not_overlapping and horiz_gap < 100:
                merged_coords = (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))
                merged_lang = lang1 or lang2
                merged_type = type1 or type2

                parts[i] = (merged_coords, (merged_type, merged_lang))
                parts.pop(i + 1)  # Safely remove the next part
                continue  # Stay at current index to recheck the merged item

            i += 1
        return parts

    @staticmethod
    def iou(b1, b2):
        """
        This function is used to calculate the intersection over union (IoU) of two bounding boxes.

        :param b1: the first bounding box
        :param b2: the second bounding box
        :return: IOU of the two bounding boxes
        """
        x1, y1, x2, y2 = b1
        x3, y3, x4, y4 = b2

        # calculate the area of the two bounding boxes
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)

        # get the intersection coordinates
        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        # calculate the area of the intersection
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

        # calculate the area of the union
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def get_region_info(self, id1, id2):
        """
        This function is used to get the language and type of the prescription from the class ids.

        :param id1: the first class id
        :param id2: the second class id
        :return: language and prescription type of the region.
        """
        # get class names from the ids
        class_name1 = list(self.classes.keys())[int(id1)]
        class_name2 = list(self.classes.keys())[int(id2)]
        # extract language and whether the region has frequency or medication
        lang = None
        prescription_type = None
        if id1 <= 1:
            lang = class_name1
        else:
            prescription_type = class_name1

        if id2 <= 1:
            lang = class_name2
        else:
            prescription_type = class_name2

        return lang, prescription_type

    def download_model(self):
        """
        Downloads the YOLO model from Google Drive.
        It creates the necessary directories and downloads the model file.
        """
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        model_link = 'https://drive.google.com/file/d/1O1gQBJUdXgEw6Ttu7p_pi63--iQVTozN/view?usp=sharing'
        gdown.download(model_link, self.local_path, quiet=False, fuzzy=True)
