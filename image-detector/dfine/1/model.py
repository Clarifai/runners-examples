# Standard library imports
import os
from typing import List, Dict, Any, Iterator

# Third-party imports
import torch
from PIL import Image as PILImage
from transformers import DFineForObjectDetection, AutoImageProcessor
from torchvision.ops import nms

# Clarifai imports
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.visual_detector_class import VisualDetectorClass
from clarifai.runners.utils.data_types import Image, Video, Region, Frame, Concept
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger


def detect_objects(
    images: List[PILImage.Image],
    model: DFineForObjectDetection,
    processor: AutoImageProcessor,
    device: str,
    threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """Process images through the D-Fine model to detect objects.

    Args:
        images: List of preprocessed PIL images
        model: D-Fine model instance
        processor: Image processor for D-Fine
        device: Computation device (CPU/GPU)
        threshold: Confidence threshold for detections

    Returns:
        Detection results from the model
    """
    model_inputs = processor(images=images, return_tensors="pt").to(device)
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    model_output = model(**model_inputs)
    results = processor.post_process_object_detection(model_output, threshold=threshold)
    return results


def process_detections_with_nms(
    results: List[Dict[str, torch.Tensor]],
    model_labels: Dict[int, str],
    iou_threshold: float = 0.2,
    use_nms: bool = True
) -> List[List[Region]]:
    """Convert model outputs into a structured format of detections, with optional NMS.

    Args:
        results: Raw detection results from model
        model_labels: Dictionary mapping label indices to names
        iou_threshold: IoU threshold for non-maximum suppression
        use_nms: Whether to apply non-maximum suppression

    Returns:
        List of lists containing Region objects for each detection
    """
    outputs = []
    for result in results:
        detections = []

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        # Apply NMS if enabled
        if use_nms and len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        for score, label_idx, box in zip(scores, labels, boxes):
            label = model_labels[label_idx.item()]
            detections.append(
                Region(
                    box=box.tolist(),
                    concepts=[Concept(id=label, name=label, value=score.item())]
                )
            )

        outputs.append(detections)

    return outputs


class MyRunner(VisualDetectorClass):
    """A custom runner for D-Fine object detection model that processes images and videos."""

    def load_model(self):
        """Load the D-Fine model and processor."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        checkpoint_path = builder.download_checkpoints(stage="runtime")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running on device: {self.device}")

        self.model = DFineForObjectDetection.from_pretrained(checkpoint_path).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model_labels = self.model.config.id2label

        logger.info("Done loading D-Fine model!")

    @VisualDetectorClass.method
    def predict(
        self,
        image: Image,
        threshold: float = Param(
            default=0.25,
            min_value=0.,
            max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid.",
        ),
        use_nms: bool = Param(
            default=True,
            description="Enable non-maximum suppression to reduce overlapping detections.",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        )
    ) -> List[Region]:
        """Process a single image and return detected objects."""
        pil_image = VisualDetectorClass.preprocess_image(image.bytes)

        with torch.no_grad():
            results = detect_objects(
                [pil_image], self.model, self.processor, self.device, threshold=threshold
            )
            outputs = process_detections_with_nms(
                results, self.model_labels, iou_threshold=iou_threshold, use_nms=use_nms
            )
            return outputs[0]

    @VisualDetectorClass.method
    def generate(
        self,
        video: Video,
        threshold: float = Param(
            default=0.25,
            min_value=0.,
            max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid.",
        ),
        use_nms: bool = Param(
            default=True,
            description="Enable non-maximum suppression to reduce overlapping detections.",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        )
    ) -> Iterator[Frame]:
        """Process video frames and yield detected objects for each frame."""
        frame_generator = VisualDetectorClass.video_to_frames(video.bytes)
        for frame in frame_generator:
            with torch.no_grad():
                pil_image = VisualDetectorClass.preprocess_image(frame.image.bytes)
                results = detect_objects(
                    [pil_image], self.model, self.processor, self.device, threshold=threshold
                )
                outputs = process_detections_with_nms(
                    results, self.model_labels, iou_threshold=iou_threshold, use_nms=use_nms
                )
                frame.regions = outputs[0]
                yield frame

    @VisualDetectorClass.method
    def stream_image(
        self,
        image_stream: Iterator[Image],
        threshold: float = Param(
            default=0.25,
            min_value=0.,
            max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid.",
        ),
        use_nms: bool = Param(
            default=True,
            description="Enable non-maximum suppression to reduce overlapping detections.",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        )
    ) -> Iterator[List[Region]]:
        """Stream process image inputs."""
        for image in image_stream:
            result = self.predict(
                image, threshold=threshold, use_nms=use_nms, iou_threshold=iou_threshold
            )
            yield result

    @VisualDetectorClass.method
    def stream_video(
        self,
        video_stream: Iterator[Video],
        threshold: float = Param(
            default=0.25,
            min_value=0.,
            max_value=1.,
            description="Minimum confidence score required for a detection to be considered valid.",
        ),
        use_nms: bool = Param(
            default=True,
            description="Enable non-maximum suppression to reduce overlapping detections.",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        )
    ) -> Iterator[Frame]:
        """Stream process video inputs."""
        for video in video_stream:
            for frame_result in self.generate(
                video, threshold=threshold, use_nms=use_nms, iou_threshold=iou_threshold
            ):
                yield frame_result

    def test(self):
        """Test the model functionality."""
        import requests

        TEST_URLS = {
            "images": [
                "https://samples.clarifai.com/metro-north.jpg",
                "https://samples.clarifai.com/dog.tiff"
            ],
            "video": "https://samples.clarifai.com/beer.mp4"
        }

        def get_test_image(url):
            return Image(bytes=requests.get(url).content)

        def get_test_video():
            return Video(bytes=requests.get(TEST_URLS["video"]).content)

        def run_test(name, test_fn):
            logger.info(f"\nTesting {name}...")
            try:
                test_fn()
                logger.info(f"{name} test completed successfully")
            except Exception as e:
                logger.error(f"Error in {name} test: {e}")

        def test_predict():
            result = self.predict(get_test_image(TEST_URLS["images"][0]))
            logger.info(f"Predict result: {result}")

        def test_generate():
            for frame in self.generate(get_test_video()):
                logger.info(f"First frame detections: {frame.regions}")
                break

        def test_stream():
            def test_stream_image():
                images = [get_test_image(url) for url in TEST_URLS["images"]]
                for result in self.stream_image(iter(images)):
                    logger.info(f"Image stream result: {result}")

            def test_stream_video():
                for frame in self.stream_video(iter([get_test_video()])):
                    logger.info(f"Video stream result: {frame.regions}")
                    break

            logger.info("\nTesting image streaming...")
            test_stream_image()
            logger.info("\nTesting video streaming...")
            test_stream_video()

        for test_name, test_fn in [
            ("predict", test_predict),
            ("generate", test_generate),
            ("stream", test_stream)
        ]:
            run_test(test_name, test_fn)
