import os
from sentence_transformers import SentenceTransformer

from config import app_settings
from utils.singelton_utils import singleton

@singleton
class ModelService:
    def __init__(self):
        self.full_model_path = app_settings.EMBEDDING_MODEL_PATH
        self.model = None

    def get_Sentence_trancsformer_model(self, refresh : bool = False)->SentenceTransformer:
        if self.model and not refresh:
            return self.model
        if os.path.exists(self.full_model_path) and os.path.isdir(self.full_model_path):
            self.model = SentenceTransformer(self.full_model_path, device=app_settings.DEVICE)
            self.model.save(self.full_model_path)
        else:
            self.model = SentenceTransformer(app_settings.EMBEDDING_MODEL, device=app_settings.DEVICE)
        return self.model