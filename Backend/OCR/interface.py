import os
import gdown
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class RegionOCR:
    def __init__(self):
        """
        Initializes the PrescriptionOCR class.
        """
        self.local_path = './Backend/Resources/weights/arabic_ocr_model'
        if not os.path.exists(self.local_path):
            self.download_arabic_model()

        # Load the OCR models here
        self.english_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.english_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

        self.arabic_processor = TrOCRProcessor.from_pretrained(self.local_path)
        self.arabic_model = VisionEncoderDecoderModel.from_pretrained(self.local_path)

    def extract_text(self, image, lang) -> str:
        """
        Extracts text from the given image.

        :param: image (str): The image to extract info from.

        Returns:
            str: The extracted text.
        """
        if lang == 'eng':
            return self.predict_eng(image)
        elif lang == 'ara':
            return self.predict_ara(image)

    def predict_ara(self, image):
        pixel_values = self.arabic_processor(image, return_tensors="pt").pixel_values
        ids = self.arabic_model.generate(pixel_values)
        return self.arabic_processor.batch_decode(ids, skip_special_tokens=True)

    def predict_eng(self, image):
        pixel_values = self.english_processor(image, return_tensors="pt").pixel_values
        ids = self.english_model.generate(pixel_values)
        return self.english_processor.batch_decode(ids, skip_special_tokens=True)

    def download_arabic_model(self):
        """
        Downloads the Arabic OCR model if it does not exist locally.
        """
        os.makedirs(self.local_path, exist_ok=True)

        # Define the model files ID's
        config_id = "1-BK33Dkt0QYzJ1wgTNkbFf0DNxiv0thc"
        gen_config_id = "1-BJndmq2OOz031-PIamHDU96FzcCnqdZ"
        model_tensor_id = "1-FOUIGy1TUaia-hnNlAZPpk_nod3LIb4"
        processor_id = "1-TEIw8ayGZubiPe5Z_g11mN6Gpqdx4A0"
        sentence_piece_id = "1-M1q7JzFoFtvJUiMWfX3rWdpq5tK_p2G"
        special_tokens_id = "1-LFe9OI_LPBAPhlXf5FTOaIcCqPaD9Un"
        tokenizer_config_id = "1-FuGqE-l7x_Wfnijr54ZBp1yA7mtU7oa"
        tokenizer_id = "1-YEsCN7RGTdc97VldafCnkrs0LG6Ib6O"

        # Download the model files
        gdown.download(f"https://drive.google.com/uc?id={config_id}",
                       os.path.join(self.local_path, 'config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={gen_config_id}",
                       os.path.join(self.local_path, 'generation_config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={model_tensor_id}",
                       os.path.join(self.local_path, 'model.safetensors'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={processor_id}",
                       os.path.join(self.local_path, 'processor_config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={sentence_piece_id}",
                       os.path.join(self.local_path, 'sentencepiece.bpe.model'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={special_tokens_id}",
                       os.path.join(self.local_path, 'special_tokens_map.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={tokenizer_config_id}",
                       os.path.join(self.local_path, 'tokenizer_config.json'),
                       quiet=False,
                       fuzzy=True)
        gdown.download(f"https://drive.google.com/uc?id={tokenizer_id}",
                       os.path.join(self.local_path, 'tokenizer.json'),
                       quiet=False,
                       fuzzy=True)
