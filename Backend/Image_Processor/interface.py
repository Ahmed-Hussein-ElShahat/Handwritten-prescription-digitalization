from typing import Any

from Backend.Roi_Extractor import RoiExtractor

import os
import cv2
import gdown
import pybktree
import Levenshtein  # for distance function


class PrescriptionOCR:
    def __init__(self):
        self.roi_extractor = RoiExtractor()
        self.frequency_corpus_url = "https://drive.google.com/file/d/1HAGoD0C8PnyjrpTdONfZ_p2NbgdGkaoA/view?usp=drive_link"
        self.medicine_corpus_url = "https://drive.google.com/file/d/1JIKGrTHapxYicaqPHImJh74Vbt2h1the/view?usp=sharing"

        self.download_corpus()
        medicines, frequencies = self.read_wordlist()

        self.medicine_BK_tree = pybktree.BKTree(Levenshtein.distance, medicines)
        self.frequency_BK_tree = pybktree.BKTree(Levenshtein.distance, frequencies)

    def process_image(self, img_path: str) -> str:
        """
        Read the image and make any needed processing before extracting text from it.

        :return: preprocessed image.
        """
        img = cv2.imread(img_path)

        # Resize the image to a fixed size
        img = cv2.resize(img, (640, 640))
        # TODO: add auto-orientation and de-skewing
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
            # get the nearst right word at a maximum distance of 3
            label, prescription_type = self.get_nearst_word_type(label, prescription_type, 3)

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

    def get_nearst_word_type(self, word: str, prescription_type: str, max_distance: int = 3) -> (str, str):
        # get the nearst word to the given label from the appointment and the medicine corpus
        frequency = self.get_nearst_word(word, self.frequency_BK_tree, 3)
        medicine = self.get_nearst_word(word, self.medicine_BK_tree, 3)

        # check if the word is in the medicine or frequency corpus
        if medicine is not None and frequency is not None:
            # check which one is the nearest
            if medicine[1] < frequency[1]:
                label = medicine[0]
                prescription_type = 'Medicine'
            elif frequency[1] < medicine[1]:
                label = frequency[0]
                prescription_type = 'Frequency'
            else:  # if they are equal then check the prescription type
                label = medicine[0] if prescription_type == 'Medicine' else frequency[0]
        elif medicine is not None:
            label = medicine[0]
            prescription_type = 'Medicine'
        elif frequency is not None:
            label = frequency[0]
            prescription_type = 'Frequency'
        else:  # if not in corpus then return the word itself
            label = word
        return label, prescription_type

    def get_nearst_word(self, word: str, bk_tree, max_distance: int = 3) -> str | None:
        """
        Get the nearest word to the given word from the medicine corpus.

        :param word: The word to find the nearest word for.
        :param bk_tree: The BK-tree to search in.
        :param max_distance: The maximum distance to consider.
        :return: The nearest word.
        """
        nearest_word = bk_tree.find(word, max_distance)
        if len(nearest_word) > 0:
            return nearest_word[0]
        else:
            return None

    def download_corpus(self):
        if not os.path.exists("Backend/Resources/appointments.txt"):
            # Download the appointment corpus
            frequency_corpus_local_path = "Backend/Resources/frequency.txt"
            gdown.download(self.frequency_corpus_url, frequency_corpus_local_path, quiet=False, fuzzy=True)

        if not os.path.exists("Backend/Resources/medicine.txt"):
            # Download the medicine corpus
            medicine_corpus_local_path = "Backend/Resources/medicine.txt"
            gdown.download(self.medicine_corpus_url, medicine_corpus_local_path, quiet=False, fuzzy=True)

    def read_wordlist(self):
        """
        Read the medicine and frequency files and return a two list of words.
        """
        with open("Backend/Resources/medicine.txt", "r") as f:
            medicines = f.read().splitlines()

        with open("Backend/Resources/frequency.txt", "r") as f:
            frequencies = f.read().splitlines()
        return medicines, frequencies
