# Standard library imports
import os
from typing import List, Dict, Any, Iterator

# Third-party imports
import cv2
import torch
from PIL import Image as PILImage
from transformers import DetrForObjectDetection, DetrImageProcessor

# Clarifai imports
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.visual_detector_class import VisualDetectorClass
from clarifai.runners.utils.data_types import Image, Video, Region, Frame
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger

def detect_objects(
    images: List[PILImage],
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    device: str
) -> Dict[str, Any]:
    """Process images through the DETR model to detect objects.

    Args:
        images: List of preprocessed images
        model: DETR model instance
        processor: Image processor for DETR
        device: Computation device (CPU/GPU)

    Returns:
        Detection results from the model
    """
    model_inputs = processor(images=images, return_tensors="pt").to(device)
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    model_output = model(**model_inputs)
    results = processor.post_process_object_detection(model_output)
    return results


class MyRunner(VisualDetectorClass):
    """A custom runner for DETR object detection model that processes images and videos"""

    def load_model(self):
        """Load the model here."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        checkpoint_path = builder.download_checkpoints(stage="runtime")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running on device: {self.device}")

        self.model = DetrForObjectDetection.from_pretrained(checkpoint_path).to(self.device)
        self.processor = DetrImageProcessor.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model_labels = self.model.config.id2label

        logger.info("Done loading!")

    @VisualDetectorClass.method
    def predict(
        self, 
        image: Image, 
        threshold: float = Param(
            default=0.9, 
            min_value=0.,
            max_value=1.,
            description="This determines the minimum probability score an object detector's prediction must have to be considered a valid detection.",
            )
    ) -> List[Region]:
        """Process a single image and return detected objects."""
        image_bytes = image.bytes
        image = VisualDetectorClass.preprocess_image(image_bytes)
        
        with torch.no_grad():
            results = detect_objects([image], self.model, self.processor, self.device)
            outputs = VisualDetectorClass.process_detections(
                results, threshold, self.model_labels)
            return outputs[0]  # Return detections for single image

    @VisualDetectorClass.method
    def generate(
        self, 
        video: Video,
        threshold: float = Param(
            default=0.9,
            min_value=0.,
            max_value=1.,
            description="This determines the minimum probability score an object detector's prediction must have to be considered a valid detection.",
        )
    ) -> Iterator[Frame]:
        """Process video frames and yield detected objects for each frame."""
        frame_generator = VisualDetectorClass.video_to_frames(video.bytes)
        for frame in frame_generator:
            with torch.no_grad():
                image = VisualDetectorClass.preprocess_image(frame.image.bytes)
                results = detect_objects([image], self.model, self.processor, self.device)
                outputs = VisualDetectorClass.process_detections(results, threshold, self.model_labels)
                frame.regions = outputs[0]  # Assign detections to the frame
                yield frame  # Yield the frame with detections

    @VisualDetectorClass.method
    def stream_image(
        self, 
        image_stream: Iterator[Image],
        threshold: float = Param(
            default=0.9,
            min_value=0.,
            max_value=1.,
            description="This determines the minimum probability score an object detector's prediction must have to be considered a valid detection.",
        )
    ) -> Iterator[List[Region]]:
        """Stream process image inputs."""
        for image in image_stream:
            result = self.predict(image, threshold=threshold)
            yield result

    @VisualDetectorClass.method
    def stream_video(
        self, 
        video_stream: Iterator[Video],
        threshold: float = Param(
            default=0.9,
            min_value=0.,
            max_value=1.,
            description="This determines the minimum probability score an object detector's prediction must have to be considered a valid detection.",
        )
    ) -> Iterator[Frame]:
        """Stream process video inputs."""
        for video in video_stream:
            for frame_result in self.generate(video, threshold=threshold):
                yield frame_result
