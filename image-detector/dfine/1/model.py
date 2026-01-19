# Standard library imports
import os
import tempfile
import time
from io import BytesIO
from typing import List, Dict, Any, Iterator

# Third-party imports
import cv2
import torch
from PIL import Image as PILImage
from transformers import DFineForObjectDetection, AutoImageProcessor
from torchvision.ops import nms

# Clarifai imports
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.utils.data_types import Concept, Image, Video, Region
from clarifai.utils.logging import logger
from clarifai.runners.utils.data_utils import Param

THRESHOLD = 0.25
IOU_THRESHOLD = 0.2

def preprocess_image(image_bytes: bytes) -> PILImage:
    """Convert image bytes into RGB format suitable for model processing.

    Args:
        image_bytes: Raw image data in bytes format

    Returns:
        PIL Image object in RGB format ready for model input
    """
    return PILImage.open(BytesIO(image_bytes)).convert("RGB")


def detect_objects(
    images: List[PILImage],
    model: DFineForObjectDetection,
    processor: AutoImageProcessor,
    device: str,
    threshold:float = THRESHOLD
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
    results = processor.post_process_object_detection(model_output, threshold=threshold)
    return results



def process_detections(
    results: List[Dict[str, torch.Tensor]],
    images: List[PILImage],
    model_labels: Dict[int, str],
    iou_threshold: float = IOU_THRESHOLD,
    use_nms: bool = True
) -> List[List[Region]]:
    """Convert model outputs into a structured format of detections, with optional NMS.

    Args:
        results: Raw detection results from model
        images: Original input images
        model_labels: Dictionary mapping label indices to names
        nms_iou_threshold: IoU threshold for non-maximum suppression
        use_nms: Whether to apply non-maximum suppression

    Returns:
        List of lists containing Region objects for each detection
    """
    outputs = []
    for i, result in enumerate(results):
        image = images[i]
        detections = []

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        # Apply NMS if enabled
        if use_nms:
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


def video_to_frames(video_bytes: bytes) -> Iterator[bytes]:
    """Convert video bytes to frames.

    Args:
        video_bytes: Raw video data in bytes

    Yields:
        JPEG encoded frame data as bytes
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name
        logger.info(f"temp_video_path: {temp_video_path}")

        video = cv2.VideoCapture(temp_video_path)
        logger.info(f"video opened: {video.isOpened()}")
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            yield frame_bytes
            
        video.release()
        os.unlink(temp_video_path)


class MyRunner(ModelClass):
    """A custom runner for DETR object detection model that processes images and videos"""

    def load_model(self):
        """Load the model here."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running on device: {self.device}")
        checkpoint_path = "ustc-community/dfine-large-obj2coco-e25"
        
        self.model = DFineForObjectDetection.from_pretrained(checkpoint_path).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model_labels = self.model.config.id2label

        logger.info("Done loading!")

    @ModelClass.method 
    def predict(
        self, 
        image: Image, 
        conf_threshold: float = Param(
            description="The `confidence threshold` value specifies the minimum confidence required for a result to be accepted or considered valid.",
            default=THRESHOLD,
        ),
        use_nms: bool = Param(
            description="Enable non-maximum suppression.",
            default=True,
        ),
        iou_threshold: float = Param(
            description="IoU threshold for non-maximum suppression, only applicable when use_nms=True",
            default=IOU_THRESHOLD,
        )
    ) -> List[Region]:
        """Process a single image and return detected objects."""
        image_bytes = image.bytes
        image = preprocess_image(image_bytes)
        logger.info(f"Recieved image: {image}")
        with torch.no_grad():
            results = detect_objects([image], self.model, self.processor, self.device, threshold=conf_threshold)
            outputs = process_detections(results, [image], self.model_labels, iou_threshold=iou_threshold, use_nms=use_nms)
            return outputs[0]  # Return detections for single image


    @ModelClass.method
    def generate(
        self, 
        video: Video,
        conf_threshold: float = Param(
            description="The `confidence threshold` value specifies the minimum confidence required for a result to be accepted or considered valid.",
            default=THRESHOLD,
        ),
        use_nms: bool = Param(
            description="Enable non-maximum suppression.",
            default=True,
        ),
        iou_threshold: float = Param(
            description="IoU threshold for non-maximum suppression, only applicable when use_nms=True",
            default=IOU_THRESHOLD,
        )
    ) -> Iterator[List[Region]]:
        """Process video frames and yield detected objects for each frame."""
        video_bytes = video.bytes
        frame_generator = video_to_frames(video_bytes)
        for frame in frame_generator:
            image = preprocess_image(frame)
            with torch.no_grad():
                results = detect_objects([image], self.model, self.processor, self.device, threshold=conf_threshold, )
                outputs = process_detections(results, [image], self.model_labels, iou_threshold=iou_threshold, use_nms=use_nms)
                yield outputs[0]  # Yield detections for each frame


    @ModelClass.method
    def stream(
        self, 
        images: Iterator[Image],
        conf_threshold: float = Param(
            description="The `confidence threshold` value specifies the minimum confidence required for a result to be accepted or considered valid.",
            default=THRESHOLD,
        ),
        use_nms: bool = Param(
            description="Enable non-maximum suppression.",
            default=True,
        ),
        iou_threshold: float = Param(
            description="IoU threshold for non-maximum suppression, only applicable when use_nms=True",
            default=IOU_THRESHOLD,
        )
    ) -> Iterator[List[Region]]:
        """Stream process image inputs."""
        logger.info("Starting stream processing for images")
        for image in images:
            start_time = time.time()
            result = self.predict(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold, use_nms=use_nms)
            yield result
            logger.info(f"Processing time: {time.time() - start_time:.3f}s")

        
    def test(self):
        """Test the model functionality."""
        import requests  # Import moved here as it's only used for testing
        
        # Test configuration
        TEST_URLS = {
            "images": [
                "https://samples.clarifai.com/metro-north.jpg",
                "https://samples.clarifai.com/dog.tiff"
            ],
            "video": "https://samples.clarifai.com/beer.mp4"
        }

        def get_test_data(url):
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

        # Test predict
        def test_predict():
            result = self.predict(get_test_data(TEST_URLS["images"][0]))
            logger.info(f"Predict result: {result}")

        # Test generate
        def test_generate():
            for detections in self.generate(get_test_video()):
                logger.info(f"First frame detections: {detections}")
                break

        # Test stream
        def test_stream():
            # Split into two separate test functions for clarity
            def test_stream_image():
                images = [get_test_data(url) for url in TEST_URLS["images"]]
                for result in self.stream(iter(images)):
                    logger.info(f"Image stream result: {result}")

            logger.info("\nTesting image streaming...")
            test_stream_image()

        # Run all tests
        for test_name, test_fn in [
            ("predict", test_predict),
            ("generate", test_generate),
            ("stream", test_stream)
        ]:
            run_test(test_name, test_fn)
