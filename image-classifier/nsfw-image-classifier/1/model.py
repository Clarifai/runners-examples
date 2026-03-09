import os
from typing import List, Iterator

# Third-party imports
import cv2
import torch
from PIL import Image as PILImage
from transformers import AutoModelForImageClassification, ViTImageProcessor

# Clarifai imports
from clarifai.runners.models.visual_classifier_class import VisualClassifierClass
from clarifai.runners.utils.data_types import Concept, Image, Video
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.utils.logging import logger


class ImageClassifierModel(VisualClassifierClass):
    """A custom runner that classifies images and outputs concepts."""

    def load_model(self):
        """Load the model and processor."""

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        checkpoints = builder.download_checkpoints(stage="runtime")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running on device: {self.device}")

        self.model = AutoModelForImageClassification.from_pretrained(checkpoints,).to(self.device)
        self.model_labels = self.model.config.id2label
        self.processor = ViTImageProcessor.from_pretrained(checkpoints)

        logger.info("Done loading!")

    @VisualClassifierClass.method
    def predict(self, image: Image) -> List[Concept]:
        """Predict concepts for a list of images."""
        pil_image = VisualClassifierClass.preprocess_image(image.bytes)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        outputs = VisualClassifierClass.process_concepts(logits, self.model_labels)
        return outputs[0]

    @VisualClassifierClass.method
    def generate(self, video: Video) -> Iterator[List[Concept]]:
        """Generate concepts for frames extracted from a video."""
        video_bytes = video.bytes
        frame_generator = VisualClassifierClass.video_to_frames(video_bytes)
        for frame in frame_generator:
            image = VisualClassifierClass.preprocess_image(frame.image.bytes)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                outputs = VisualClassifierClass.process_concepts(logits, self.model_labels)  # Yield concepts for each frame
                yield outputs[0]

    @VisualClassifierClass.method
    def stream_image(self, image_stream: Iterator[Image]) -> Iterator[List[Concept]]:
        """Stream process image inputs."""
        for image in image_stream:
            result = self.predict(image)
            yield result

    @VisualClassifierClass.method
    def stream_video(self, video_stream: Iterator[Video]) -> Iterator[List[Concept]]:
        """Stream process video inputs."""
        for video in video_stream:
            for frame_result in self.generate(video):
                yield frame_result