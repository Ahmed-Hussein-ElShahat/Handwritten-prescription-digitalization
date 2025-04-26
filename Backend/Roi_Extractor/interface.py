import os.path

import gdown
import cv2
from ultralytics import YOLO

from Backend.OCR import RegionOCR


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

        # Load the OCR model here
        self.ocr = RegionOCR()

    def extract_info(self, img):
        """
        Predicts the regions of interest in the given image using the YOLO model
        and gets the description associated with each one.

        :param img: the image which should be processed
        :return: parts of the image each one include the bounding box, the description of it and the predicted label
        """

        boxes = self.get_ROI(img)
        return self.add_lang_type(boxes, img)

    def get_ROI(self, img):
        """
        This function is used to get the region of interest (ROI) from the image using the YOLO model.

        :param img: the image which should be processed
        :return: bounding boxes, confidence scores and class ids
        """
        # get the region of interest (ROI) from the image
        roi = self.yolo.predict(img, conf=0.5)

        result = roi[0].obb
        result = sorted(result, key=lambda x: x.xyxy[0][1].numpy())  # sort according to the y-axis

        # get the bounding boxes, confidence scores and class ids
        conf = []
        xyxy = []
        cls_ids = []
        for res in result:
            cls_ids.append(int(res.cls[0].numpy()))
            conf.append(res.conf[0].numpy())
            xyxy.append(res.xyxy[0].numpy())

        return xyxy, conf, cls_ids

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

        # TODO: check if the medicine number not equal to the frequency number or if prescription doesnt have a language
        for i in range(len(bbox)):
            for j in range(i + 1, len(bbox)):
                if self.iou(bbox[i], bbox[j]) > 0.3:  # extract the info when the boxes refer to the same region
                    lang, prescription_type = self.get_region_info(cls_ids[i], cls_ids[j])

                    # crop the region of interest
                    x1, y1, x2, y2 = bbox[i]
                    cropped_img = img[y1:y2, x1:x2]

                    # get the label of the ROI
                    if prescription_type == 'Medicine':
                        # Use the OCR to get the medicine name
                        label = self.ocr.extract_text(cropped_img, lang[:3].tolower())
                        medicine_index += 1
                    else:
                        # Use the OCR to get the frequency
                        label = self.ocr.extract_text(cropped_img, lang[:3].tolower())
                        frequency_index += 1

                    # add the label with the other info into the list
                    parts.append((bbox[i], (prescription_type, lang), label))
                    break

            if frequency_index == num_freq and medicine_index == num_med:
                break

        # TODO: add the rest of the boxes which are not related to any other box

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

        return intersection_area / union_area

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
        # lang=None
        # prescription_type = None
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
