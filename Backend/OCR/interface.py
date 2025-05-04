import os
import cv2
import gdown
import cProfile
import pstats
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging
import re

class RegionOCR:
    def __init__(self):
        """
        Initializes the PrescriptionOCR class.
        """
        self.local_path_arabic_model = './Backend/Resources/weights/arabic_ocr_model'
        self.loacl_path_english_model = './Backend/Resources/weights/english_ocr_model'

        self.download_arabic_model()
        self.download_english_model()

        # suppress the unnecessary messages during inference
        logging.set_verbosity_error()

        # Load the OCR models here
        # load the arabic model
        self.arabic_processor = TrOCRProcessor.from_pretrained(self.local_path_arabic_model)
        self.arabic_model = VisionEncoderDecoderModel.from_pretrained(self.local_path_arabic_model)

        #load the english model
        self.english_processor = TrOCRProcessor.from_pretrained(self.loacl_path_english_model)
        self.english_model = VisionEncoderDecoderModel.from_pretrained(self.loacl_path_english_model)

    def extract_text(self, image, lang) -> str:
        """
        Extracts text from the given image.

        :param: image : The image to extract info from.
        :param: lang : The language of the text in the image (0 for Arabic, 1 for English).
        Returns:
            str: The extracted text.
        """

        if lang == 1:
            text = self.predict_eng(image)[0]
        elif lang == 0:
            text = self.predict_ara(image)[0]
        else:
            return ''

        label = re.sub(r"[^\w\s]", "", text)
        label = label.strip()
        return label

    def predict_ara(self, image):
        pixel_values = self.arabic_processor(image, return_tensors="pt").pixel_values
        ids = self.arabic_model.generate(pixel_values)
        return self.arabic_processor.batch_decode(ids, skip_special_tokens=True)

    def predict_eng(self, image):
        pixel_values = self.english_processor(image, return_tensors="pt").pixel_values
        ids = self.english_model.generate(pixel_values)
        return self.english_processor.batch_decode(ids, skip_special_tokens=True)

    def download_english_model(self):
        """
        Downloads the English OCR model if it does not exist locally.
        """
        # path to the created .complete file
        path_to_complete_file = os.path.join(self.loacl_path_english_model, '.complete')
        if os.path.exists(path_to_complete_file):
          return

        # if it oesnt exist then make directory and download the model in it
        os.makedirs(self.loacl_path_english_model, exist_ok=True)

        # ge the model link then download the model
        model_link = 'https://drive.google.com/drive/folders/1-9_Cz6xnVMab6IjEPrXf8zWLf5H7cion?usp=drive_link'
        gdown.download_folder(model_link, output=self.loacl_path_english_model, quiet=False)

        # make a file to tell that the model has been downloaded without it the model is treated as not downloaded
        with open(os.path.join(self.loacl_path_english_model, '.complete'), 'w') as f:
          f.write('true')

    def download_arabic_model(self):
        """
        Downloads the Arabic OCR model if it does not exist locally.
        """

        # check if the model already exist
        # path to the created .complete file
        path_to_complete_file = os.path.join(self.local_path_arabic_model, '.complete')
        if os.path.exists(path_to_complete_file):
          return

        os.makedirs(self.local_path_arabic_model, exist_ok=True)

        # Define the model files ID's
        config_id = "1-QuP1ZoXsAjJSkO2AVrXiSIdUhp3zww_"
        gen_config_id = "1-ORYC4N1liybcyuDihjGSNzWqV3oSH9J"
        model_tensor_id = "1-YlWSsZy0TGHwtU2eR7Ng78k0dYJuKm0"
        processor_id = "1-NjWDOpHa9QHh_W5xdu-u_pfoufj7-1K"
        sentence_piece_id = "1-WYclVqXwNxXMfRE5tN6K-VVjkPzyQq6"
        special_tokens_id = "1-HCqIsfz56dg0V-fBs8VzGVs92sPZg8L"
        tokenizer_config_id = "1-B65XNidxQNdZxW8QijJiQUmru6oHz9K"
        tokenizer_id = "1-8dheZMmlG879XGsGJiMgpvgNZot6QtN"

        # Download the model files
        gdown.download(f"https://drive.google.com/uc?id={config_id}",
                       os.path.join(self.local_path_arabic_model, 'config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={gen_config_id}",
                       os.path.join(self.local_path_arabic_model, 'generation_config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={model_tensor_id}",
                       os.path.join(self.local_path_arabic_model, 'model.safetensors'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={processor_id}",
                       os.path.join(self.local_path_arabic_model, 'preprocessor_config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={sentence_piece_id}",
                       os.path.join(self.local_path_arabic_model, 'sentencepiece.bpe.model'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={special_tokens_id}",
                       os.path.join(self.local_path_arabic_model, 'special_tokens_map.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={tokenizer_config_id}",
                       os.path.join(self.local_path_arabic_model, 'tokenizer_config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={tokenizer_id}",
                       os.path.join(self.local_path_arabic_model, 'tokenizer.json'),
                       quiet=False,
                       fuzzy=True)

        # make a file to tell that the model has been downloaded without it the model is treated as not downloaded
        with open(os.path.join(self.local_path_arabic_model, '.complete'), 'w') as f:
          f.write('true')
