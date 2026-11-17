from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tensorflow import keras


class ClassifierService:
    def __init__(self, model_dir: Path, dimension: int = 1024, num_classes: int = 100):
        self.model_dir = Path(model_dir)
        self.dimension = dimension
        self.num_classes = num_classes
        self.model: Any = None
        self.scaler: Any = None
        self.class_names: dict[int, str] = {}

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None and len(self.class_names) == self.num_classes

    def load(self) -> None:
        required = ["classifier.keras", "scaler.joblib", "class_names.json", "config.json"]
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing classifier artifacts in {self.model_dir}: {', '.join(missing)}")
        config = json.loads((self.model_dir / "config.json").read_text())
        if config.get("embedding_model", "").lower() != "baai/bge-m3":
            raise ValueError("Classifier config is not for BAAI/bge-m3")
        if config.get("embedding_dimension") != self.dimension or config.get("num_classes") != self.num_classes:
            raise ValueError("Classifier configuration dimensions do not match runtime settings")
        model = keras.models.load_model(self.model_dir / "classifier.keras", compile=False)
        scaler = joblib.load(self.model_dir / "scaler.joblib")
        raw_names = json.loads((self.model_dir / "class_names.json").read_text())
        class_names = {int(key): str(value) for key, value in raw_names.items()} if isinstance(raw_names, dict) else {i: str(v) for i, v in enumerate(raw_names)}
        if tuple(model.input_shape[1:]) != (self.dimension,) or tuple(model.output_shape[1:]) != (self.num_classes,):
            raise ValueError("Classifier input/output dimensions do not match configured dimensions")
        if getattr(scaler, "n_features_in_", None) != self.dimension:
            raise ValueError("Fitted scaler input dimension does not match configured dimension")
        if len(class_names) != self.num_classes:
            raise ValueError("Class mapping size does not match configured class count")
        self.model, self.scaler, self.class_names = model, scaler, class_names

    def predict_embedding(self, embedding: list[float]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Classifier is not loaded")
        values = np.asarray(embedding, dtype=np.float32)
        if values.shape != (self.dimension,):
            raise ValueError(f"Expected {self.dimension}-dimensional embedding")
        scaled = self.scaler.transform(values.reshape(1, -1))
        probabilities = np.asarray(self.model.predict(scaled, verbose=0))[0]
        top_ids = np.argsort(probabilities)[-5:][::-1]
        class_id = int(top_ids[0])
        return {
            "predicted_class_id": class_id,
            "predicted_class": self.class_names[class_id],
            "confidence": float(probabilities[class_id]),
            "top_predictions": [
                {"class_id": int(i), "class_name": self.class_names[int(i)], "confidence": float(probabilities[i])}
                for i in top_ids
            ],
        }
