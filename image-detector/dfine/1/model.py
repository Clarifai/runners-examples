# Standard library imports
import os
import signal
import sys
from io import BytesIO
from typing import List, Dict, Any, Iterator, Optional, Tuple

# Third-party imports
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import DFineForObjectDetection, AutoImageProcessor
from torchvision.ops import nms

# Clarifai imports
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.visual_detector_class import VisualDetectorClass
from clarifai.runners.utils.data_types import Image, Video, Region, Frame, Concept
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger

# TensorRT imports (optional)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None


def signal_handler(sig, frame):
    """Handle SIGINT and SIGTERM signals for graceful shutdown."""
    logger.info("\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class TensorRTInference:
    """TensorRT inference engine wrapper for D-FINE model."""

    def __init__(self, engine_path: str):
        """Initialize TensorRT inference engine.

        Args:
            engine_path: Path to the TensorRT engine file
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Install tensorrt package.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None

        # Load engine
        logger.info(f"Loading TensorRT engine from {engine_path}...")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = torch.cuda.Stream()

        # Get binding info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(1, self.engine.num_io_tensors)
        ]

        # Get input shape (for reference)
        input_shape = self.engine.get_tensor_shape(self.input_name)
        self.channels = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]

        logger.info(f"TensorRT engine loaded: input shape = {input_shape}")
        logger.info(f"Output tensors: {self.output_names}")

    def __call__(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on input tensor.

        Args:
            pixel_values: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of (logits, pred_boxes) tensors
        """
        batch_size = pixel_values.shape[0]

        # Ensure input is on GPU and contiguous
        if not pixel_values.is_cuda:
            pixel_values = pixel_values.cuda()
        pixel_values = pixel_values.contiguous()

        # Set input shape for dynamic batch
        self.context.set_input_shape(self.input_name, pixel_values.shape)

        # Allocate output buffers
        outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            # Handle dynamic dimensions
            shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda").contiguous()

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, pixel_values.data_ptr())
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Run inference
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Return logits and pred_boxes
        logits = outputs.get("logits")
        pred_boxes = outputs.get("pred_boxes")

        return logits, pred_boxes

    def __del__(self):
        """Cleanup resources."""
        self.context = None
        self.engine = None


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: float = 4.0,
) -> str:
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to input ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        workspace_gb: Workspace size in GB

    Returns:
        Path to the built engine
    """
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available")

    logger.info(f"Building TensorRT engine from {onnx_path}...")
    logger.info("This may take several minutes on first run...")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # Parse ONNX
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    logger.info("ONNX model parsed successfully")

    # Configure builder
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if fp16 and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)

    # Set up dynamic shape optimization profile
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Dynamic batch: min=1, opt=1, max=8
    profile.set_shape(input_name, (1, channels, height, width), (1, channels, height, width), (8, channels, height, width))
    config.add_optimization_profile(profile)

    logger.info(f"Building engine with input shape: (1-8, {channels}, {height}, {width})")

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved to {engine_path}")
    return engine_path


def center_to_corners_format(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center format (cx, cy, w, h) to corners format (x1, y1, x2, y2).

    Args:
        boxes: Tensor of shape (..., 4) in (cx, cy, w, h) format

    Returns:
        Tensor of shape (..., 4) in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def detect_objects_tensorrt(
    images: List[PILImage.Image],
    trt_engine: TensorRTInference,
    processor: AutoImageProcessor,
    id2label: Dict[int, str],
    threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """Process images through TensorRT engine to detect objects.

    Args:
        images: List of PIL images to process
        trt_engine: TensorRT inference engine
        processor: Image processor for preprocessing
        id2label: Mapping from class indices to labels
        threshold: Confidence threshold for detections

    Returns:
        List of detection results per image
    """
    # Preprocess images
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda()

    # Run TensorRT inference
    logits, pred_boxes = trt_engine(pixel_values)

    # Post-process results
    results = []
    batch_size = logits.shape[0]

    for i in range(batch_size):
        # Get image size for denormalization
        img = images[i]
        img_width, img_height = img.size

        # Get predictions for this image
        img_logits = logits[i]  # (num_queries, num_classes)
        img_boxes = pred_boxes[i]  # (num_queries, 4) in [cx, cy, w, h] normalized

        # Apply softmax to get probabilities (exclude background class if present)
        probs = torch.softmax(img_logits, dim=-1)

        # Get max probability and class for each query (excluding background)
        # D-FINE uses last class as background
        scores, labels = probs[:, :-1].max(dim=-1)

        # Filter by threshold
        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = img_boxes[keep]

        # Convert from center format to corners format
        boxes = center_to_corners_format(boxes)

        # Denormalize boxes to pixel coordinates
        boxes[:, [0, 2]] *= img_width
        boxes[:, [1, 3]] *= img_height

        results.append({
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        })

    return results


def detect_objects(
    images: List[PILImage.Image],
    model: DFineForObjectDetection,
    processor: AutoImageProcessor,
    device: str,
    threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """Process images through the D-Fine model to detect objects."""
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
    """Convert model outputs into a structured format of detections, with optional NMS."""
    outputs = []
    for result in results:
        detections = []

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

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

    def __init__(self):
        super().__init__()
        self._model: Optional[DFineForObjectDetection] = None
        self._processor: Optional[AutoImageProcessor] = None
        self._model_labels: Optional[Dict[int, str]] = None
        self._device: Optional[str] = None
        self._checkpoint_path: Optional[str] = None
        self._trt_engine: Optional[TensorRTInference] = None
        self._use_tensorrt: bool = False

    def load_model(self):
        """Download checkpoints and initialize model/TensorRT engine."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        self._checkpoint_path = builder.download_checkpoints(stage="runtime")
        logger.info(f"Checkpoints ready at: {self._checkpoint_path}")

        # Eagerly load model/TensorRT engine at startup
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Lazy load model on first use - called from worker process after fork."""
        if self._model is not None or self._trt_engine is not None:
            return

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing model on device: {self._device}")

        # Paths for TensorRT engine and ONNX model
        engine_path = os.path.join(self._checkpoint_path, "dfine.engine")
        onnx_path = os.path.join(self._checkpoint_path, "dfine.onnx")

        # Also check for ONNX in the model directory (alongside model.py)
        model_dir = os.path.dirname(__file__)
        model_dir_onnx = os.path.join(model_dir, "dfine.onnx")
        if not os.path.exists(onnx_path) and os.path.exists(model_dir_onnx):
            onnx_path = model_dir_onnx

        # Try TensorRT if available and on CUDA
        if TENSORRT_AVAILABLE and self._device == 'cuda':
            # Build TensorRT engine from ONNX if not exists
            if not os.path.exists(engine_path) and os.path.exists(onnx_path):
                try:
                    logger.info("TensorRT engine not found, building from ONNX...")
                    build_tensorrt_engine(onnx_path, engine_path, fp16=True)
                except Exception as e:
                    logger.warning(f"Failed to build TensorRT engine: {e}")

            # Try to load the engine
            if os.path.exists(engine_path):
                try:
                    self._trt_engine = TensorRTInference(engine_path)
                    self._use_tensorrt = True
                    logger.info("TensorRT engine loaded successfully!")

                    # Still need processor and labels from PyTorch model config
                    self._processor = AutoImageProcessor.from_pretrained(self._checkpoint_path, use_fast=True)

                    # Load config for id2label mapping
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(self._checkpoint_path)
                    self._model_labels = config.id2label

                    logger.info("D-Fine model ready with TensorRT backend!")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load TensorRT engine: {e}")
                    logger.info("Falling back to PyTorch backend...")
                    self._trt_engine = None
                    self._use_tensorrt = False

        # Fallback to PyTorch model
        logger.info("Loading PyTorch model...")
        self._model = DFineForObjectDetection.from_pretrained(self._checkpoint_path).to(self._device)
        self._processor = AutoImageProcessor.from_pretrained(self._checkpoint_path, use_fast=True)
        self._model.eval()
        self._model_labels = self._model.config.id2label

        logger.info("D-Fine model loaded successfully with PyTorch backend!")

    def _detect_objects(
        self,
        images: List[PILImage.Image],
        threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """Run object detection using the appropriate backend (TensorRT or PyTorch).

        Args:
            images: List of PIL images to process
            threshold: Confidence threshold for detections

        Returns:
            List of detection results per image
        """
        if self._use_tensorrt and self._trt_engine is not None:
            return detect_objects_tensorrt(
                images,
                self._trt_engine,
                self._processor,
                self._model_labels,
                threshold=threshold
            )
        else:
            return detect_objects(
                images,
                self._model,
                self._processor,
                self._device,
                threshold=threshold
            )

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
            default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based).",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        )
    ) -> List[Region]:
        """Process a single image and return detected objects."""
        self._ensure_model_loaded()
        pil_image = PILImage.open(BytesIO(image.bytes)).convert("RGB")

        with torch.no_grad():
            results = self._detect_objects([pil_image], threshold=threshold)
            outputs = process_detections_with_nms(
                results, self._model_labels, iou_threshold=iou_threshold, use_nms=use_nms
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
            default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based).",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        )
    ) -> Iterator[Frame]:
        """Process video frames and yield detected objects for each frame."""
        self._ensure_model_loaded()
        frame_generator = VisualDetectorClass.video_to_frames(video.bytes)
        for frame in frame_generator:
            with torch.no_grad():
                pil_image = PILImage.open(BytesIO(frame.image.bytes)).convert("RGB")
                results = self._detect_objects([pil_image], threshold=threshold)
                outputs = process_detections_with_nms(
                    results, self._model_labels, iou_threshold=iou_threshold, use_nms=use_nms
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
            default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based).",
        ),
        iou_threshold: float = Param(
            default=0.2,
            min_value=0.,
            max_value=1.,
            description="IoU threshold for non-maximum suppression (only used when use_nms=True).",
        ),
        batch_size: int = Param(
            default=1,
            min_value=1,
            max_value=32,
            description="Number of images to batch together. Use 1 for lowest latency (real-time streaming), or higher (4-8) for maximum throughput (offline processing).",
        )
    ) -> Iterator[List[Region]]:
        """Stream process image inputs.

        For real-time streaming (live video): Use batch_size=1 for lowest latency.
        For offline processing (maximum throughput): Use batch_size=4-8.

        Batching increases throughput but adds latency in streaming scenarios.
        """
        self._ensure_model_loaded()

        batch = []
        for image in image_stream:
            pil_image = PILImage.open(BytesIO(image.bytes)).convert("RGB")
            batch.append(pil_image)

            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                with torch.no_grad():
                    results = self._detect_objects(batch, threshold=threshold)
                    outputs = process_detections_with_nms(
                        results, self._model_labels, iou_threshold=iou_threshold, use_nms=use_nms
                    )
                    # Yield each result individually
                    for output in outputs:
                        yield output
                batch = []

        # Process remaining images in the last partial batch
        if batch:
            with torch.no_grad():
                results = self._detect_objects(batch, threshold=threshold)
                outputs = process_detections_with_nms(
                    results, self._model_labels, iou_threshold=iou_threshold, use_nms=use_nms
                )
                for output in outputs:
                    yield output

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
            default=False,
            description="Enable non-maximum suppression to reduce overlapping detections. Not needed for D-FINE (DETR-based).",
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
