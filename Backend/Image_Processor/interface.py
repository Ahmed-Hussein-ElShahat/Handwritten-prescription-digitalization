from Backend.Roi_Extractor import RoiExtractor
import cv2


class PrescriptionOCR:
    def __init__(self):
        self.roi_extractor = RoiExtractor()
        # TODO: instantiate a BK tree
    def process_image(self, img_path: str) -> str:
        """
        Read the image and make any needed processing before extracting text from it.

        :return: preprocessed image.
        """
        img = cv2.imread(img_path)

        # Resize the image to a fixed size
        img = cv2.resize(img, (640, 640))
        return img

    def predict(self, img_path: str) -> list:
        """
        Predict the text from the image.

        :return: predicted text.
        """
        # Read and process the image
        img = self.process_image(img_path)

        # Get ROI from the image
        image_parts = self.roi_extractor.extract_info(img)

        num_freq = 0
        num_med = 0
        last_type = None
        last_label = None
        pairs = []
        for part in image_parts:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = part[0]
            # Get the ROI description
            prescription_type, lang = part[1]

            # Get the text from the image
            label = part[2]

            # TODO: check if prescription type is empty
            # Check if the last type is not None
            if last_type is not None:
                # Check if the last type is different from the current type
                if last_type != prescription_type:
                    # pair is (medicine, frequency)
                    pair = (label, last_label) if prescription_type == 'Medicine' else (last_label, label)
                    pairs.append(pair)

                    # change last type to be the current type
                    last_type = prescription_type

                else:
                    last_label = label
            else:
                last_label = label
                last_type = prescription_type
        return pairs
