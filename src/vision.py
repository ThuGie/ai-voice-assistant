"""
Vision Module using PyTorch Vision

This module handles computer vision capabilities including screen capture,
webcam access, and image analysis using PyTorch Vision.
"""

import os
import sys
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Set up logging
logger = logging.getLogger(__name__)

class VisionEngine:
    """Computer Vision engine using PyTorch Vision"""
    
    # COCO dataset classes for object detection
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Vision engine.
        
        Args:
            device: Device to use for model inference ("cuda", "cpu", or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize screen capture
        self.screen_capture = None
        
        # Initialize webcam capture
        self.webcam = None
        self.webcam_id = 0
        
        try:
            self._load_models()
            logger.info(f"Initialized Vision engine on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Vision engine: {e}")
            raise
    
    def _load_models(self):
        """Load the computer vision models"""
        try:
            # Load image classification model
            logger.info("Loading image classification model...")
            self.classification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.classification_model.to(self.device)
            self.classification_model.eval()
            
            # Prepare classification transforms
            self.classification_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Load object detection model
            logger.info("Loading object detection model...")
            self.detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            
            logger.info("Vision models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vision models: {e}")
            raise
    
    def _initialize_screen_capture(self):
        """Initialize screen capture"""
        try:
            # Try to use D3DShot for faster captures on Windows
            try:
                import d3dshot
                self.screen_capture = d3dshot.create(capture_output="numpy")
                logger.info("Using D3DShot for screen capture")
                return
            except ImportError:
                logger.info("D3DShot not available, falling back to MSS")
            
            # Fall back to MSS (cross-platform)
            import mss
            self.screen_capture = mss.mss()
            logger.info("Using MSS for screen capture")
        except Exception as e:
            logger.error(f"Failed to initialize screen capture: {e}")
            raise
    
    def _initialize_webcam(self, webcam_id: int = 0):
        """
        Initialize webcam capture.
        
        Args:
            webcam_id: ID of the webcam to use
        """
        try:
            self.webcam = cv2.VideoCapture(webcam_id)
            if not self.webcam.isOpened():
                raise Exception(f"Could not open webcam with ID {webcam_id}")
            self.webcam_id = webcam_id
            logger.info(f"Initialized webcam with ID {webcam_id}")
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            raise
    
    def release_webcam(self):
        """Release the webcam if it's open"""
        if self.webcam is not None and self.webcam.isOpened():
            self.webcam.release()
            self.webcam = None
            logger.info("Released webcam")
    
    def capture_screen(self, monitor: int = 0) -> np.ndarray:
        """
        Capture the screen.
        
        Args:
            monitor: Monitor ID to capture
            
        Returns:
            Screen image as a numpy array (RGB format)
        """
        if self.screen_capture is None:
            self._initialize_screen_capture()
        
        try:
            # Handle different screen capture methods
            if hasattr(self.screen_capture, "grab"):  # D3DShot
                # D3DShot returns RGB format directly
                return self.screen_capture.grab(monitor=monitor)
            else:  # MSS
                # Get monitor information
                monitor_dict = self.screen_capture.monitors[monitor] if monitor < len(self.screen_capture.monitors) else self.screen_capture.monitors[0]
                
                # Capture screenshot
                screenshot = self.screen_capture.grab(monitor_dict)
                
                # Convert to numpy array and RGB format (from BGR)
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            # Return a black image as fallback
            return np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def capture_webcam(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the webcam.
        
        Returns:
            Webcam image as a numpy array (RGB format) or None if failed
        """
        if self.webcam is None:
            try:
                self._initialize_webcam()
            except Exception as e:
                logger.error(f"Could not initialize webcam: {e}")
                return None
        
        try:
            ret, frame = self.webcam.read()
            if not ret:
                logger.error("Failed to capture frame from webcam")
                return None
            
            # Convert from BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Webcam capture failed: {e}")
            return None
    
    def save_image(self, image: np.ndarray, filename: Optional[str] = None) -> str:
        """
        Save an image to a file.
        
        Args:
            image: Image as a numpy array
            filename: Filename to save to (optional)
            
        Returns:
            Path to the saved image
        """
        if filename is None:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            
            # Create a temporary file if no directory is specified
            if not os.path.dirname(filename):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    filename = temp_file.name
        
        try:
            # Convert to PIL Image and save
            Image.fromarray(image).save(filename)
            logger.info(f"Image saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise
    
    def classify_image(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify an image and return top-k class predictions.
        
        Args:
            image: Image as a numpy array
            top_k: Number of top classes to return
            
        Returns:
            List of (class_name, confidence) tuples
        """
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply transformations
            input_tensor = self.classification_transforms(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classification_model(input_batch)
            
            # Get top-k predictions
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Map indices to class names and return results
            results = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][idx.item()]
                confidence = prob.item()
                results.append((class_name, confidence))
            
            return results
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            return []
    
    def detect_objects(self, image: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as a numpy array
            threshold: Confidence threshold for detections
            
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Convert to tensor
            input_tensor = transforms.ToTensor()(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.detection_model(input_batch)
            
            # Process detections
            detections = []
            for i, (boxes, labels, scores) in enumerate(zip(predictions[0]['boxes'], 
                                                          predictions[0]['labels'],
                                                          predictions[0]['scores'])):
                if scores > threshold:
                    # Get coordinates
                    box = boxes.cpu().numpy()
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Get class name
                    class_idx = labels.cpu().item()
                    class_name = self.COCO_CLASSES[class_idx]
                    
                    # Get confidence
                    confidence = scores.cpu().item()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': (x1, y1, x2, y2)
                    })
            
            return detections
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Annotate an image with detection results.
        
        Args:
            image: Image as a numpy array
            detections: List of detection results from detect_objects
            
        Returns:
            Annotated image as a numpy array
        """
        # Create a copy of the image
        annotated_image = image.copy()
        
        # Convert to BGR for OpenCV drawing
        if annotated_image.shape[2] == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes and labels
        for detection in detections:
            # Get detection info
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['box']
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Convert back to RGB if needed
        if annotated_image.shape[2] == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
        return annotated_image
    
    def describe_image(self, image: np.ndarray) -> str:
        """
        Generate a description of the image.
        
        Args:
            image: Image as a numpy array
            
        Returns:
            Description of the image
        """
        # Get image classification
        classifications = self.classify_image(image)
        
        # Get object detections
        detections = self.detect_objects(image)
        
        # Generate description
        description = "I can see "
        
        # Add classification results
        if classifications:
            top_class, confidence = classifications[0]
            description += f"what appears to be {top_class} (confidence: {confidence:.2f}). "
            
            if len(classifications) > 1:
                description += "It might also be "
                for i, (class_name, conf) in enumerate(classifications[1:3]):  # Just add 2nd and 3rd possibilities
                    description += f"{class_name} (confidence: {conf:.2f})"
                    if i < len(classifications[1:3]) - 1:
                        description += " or "
                description += ". "
        
        # Add detection results
        if detections:
            description += "I've detected the following objects: "
            for i, detection in enumerate(detections):
                description += f"{detection['class']} (confidence: {detection['confidence']:.2f})"
                if i < len(detections) - 1:
                    description += ", "
            description += "."
        
        # Fallback if no information
        if not classifications and not detections:
            description = "I can't identify anything specific in this image."
            
        return description


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vision engine
    vision = VisionEngine()
    
    # Capture screen
    print("Capturing screen...")
    screen = vision.capture_screen()
    
    # Save the captured screen
    vision.save_image(screen, "screen_capture.png")
    
    # Classify the image
    print("\nClassifying image...")
    classifications = vision.classify_image(screen)
    print("Top classifications:")
    for class_name, confidence in classifications:
        print(f"- {class_name}: {confidence:.4f}")
    
    # Detect objects
    print("\nDetecting objects...")
    detections = vision.detect_objects(screen)
    print(f"Found {len(detections)} objects:")
    for detection in detections:
        print(f"- {detection['class']} ({detection['confidence']:.4f}): {detection['box']}")
    
    # Generate image description
    print("\nImage description:")
    description = vision.describe_image(screen)
    print(description)
