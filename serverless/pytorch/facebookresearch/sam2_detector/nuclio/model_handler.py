# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import logging
from PIL import Image

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Try to import cv2 for mask to polygon conversion
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # Warning about cv2 unavailability will be logged in __init__

# Define paths for model weights and configs
# These paths are relative to the workdir inside the Nuclio function container
CNN_MODEL_PATH = "./model.pt"  # Your finetuned_ssdlite320_mobilenetv3_model.pt copied as model.pt
SAM2_CHECKPOINT_PATH = "./sam2_hiera_large.pt"
SAM2_MODEL_CFG = "sam2_hiera_l.yaml" # Make sure this SAM2 config file is in your build context or accessible

class DetectorHandler:
    NUM_CLASSES = 2  # 1 (corrosion) + 1 (background)
    LABEL_MAP = {1: "corrosion"}  # From your fine-tune.ipynb (label 'corrosion' is idx 1)
    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        if not CV2_AVAILABLE:
            self.logger.warn("opencv-python is not available. SAM2 mask to polygon conversion will not work. Bounding boxes will be returned if SAM2 is used.") # Changed warning to warn

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # 1. Load CNN Detection Model (SSDLite MobileNetV3 Large)
        try:
            self.logger.info(f"Loading CNN model (SSDLite MobileNetV3) from {CNN_MODEL_PATH}...")
            # Initialize the model architecture
            self.cnn_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=self.NUM_CLASSES)
            # Load the fine-tuned state dictionary
            self.cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=self.device, weights_only=False))
            self.cnn_model.eval()  # Set to evaluation mode
            self.cnn_model.to(self.device) # Move model to device
            self.logger.info("CNN model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load CNN model: {str(e)}")
            raise # Fail initialization if CNN model can't be loaded

        # 2. Load SAM2 Model for Segmentation Refinement (Optional)
        try:
            self.logger.info(f"Loading SAM2 model from {SAM2_CHECKPOINT_PATH}...")
            # Ensure SAM2_MODEL_CFG path is correct or accessible in the container
            self.sam2_predictor = SAM2ImagePredictor(build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT_PATH, device=self.device))
            self.logger.info("SAM2 model loaded successfully.")
            self.use_sam2_for_segmentation = True
        except Exception as e:
            self.logger.warn(f"Failed to load SAM2 model: {str(e)}. Proceeding with only CNN bounding boxes.") # Changed warning to warn
            self.sam2_predictor = None
            self.use_sam2_for_segmentation = False

        # Define image transformation for CNN input
        self.cnn_transform = T.Compose([T.ToTensor()])

    def _run_cnn_inference(self, image: Image.Image):
        """
        Run inference with the loaded SSDLite MobileNetV3 model.
        """
        self.logger.info("Running CNN inference...")
        img_tensor = self.cnn_transform(image).to(self.device)

        with torch.no_grad():
            predictions = self.cnn_model([img_tensor]) # Model expects a list of images

        if not predictions:
            self.logger.info("CNN inference returned no predictions.")
            return []

        pred = predictions[0] # Get results for the first (and only) image

        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        results = []
        for box, label_id, score in zip(boxes, labels, scores):
            results.append({'box': box.tolist(),       # [x1, y1, x2, y2]
                            'label_id': int(label_id), # Integer class ID
                            'score': float(score)})    # Detection confidence

        self.logger.info(f"CNN inference found {len(results)} raw detections.")
        return results

    def _convert_mask_to_polygon(self, mask_np: np.ndarray):
        """
        Convert a binary mask (HxW NumPy array) to a list of polygon points [x1,y1,x2,y2,...].
        Requires opencv-python.
        """
        if not CV2_AVAILABLE:
            self.logger.warn("Attempted to convert mask to polygon, but opencv-python is not available.") # Changed warning to warn
            return None

        self.logger.info(f"Original mask_np before conversion - Shape: {mask_np.shape}, Dtype: {mask_np.dtype}, Min: {np.min(mask_np)}, Max: {np.max(mask_np)}")
        unique_values_before = np.unique(mask_np)
        self.logger.info(f"Original mask_np unique values (first 10): {unique_values_before[:10]}")

        # Ensure mask is 8-bit single-channel image (0 or 255)
        if mask_np.dtype != np.uint8:
            self.logger.info(f"Converting mask_np from {mask_np.dtype} to uint8.")
            mask_np = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1.0 else mask_np.astype(np.uint8)
        
        self.logger.info(f"Mask_np after conversion - Shape: {mask_np.shape}, Dtype: {mask_np.dtype}, Min: {np.min(mask_np)}, Max: {np.max(mask_np)}")
        unique_values_after = np.unique(mask_np)
        self.logger.info(f"Mask_np unique values after conversion (first 10): {unique_values_after[:10]}")
        self.logger.info(f"Mask_np non-zero count after conversion: {np.count_nonzero(mask_np)}")

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.logger.info("No contours found in the mask.")
            return None

        # Assuming the largest contour is the primary object for the mask
        largest_contour = max(contours, key=cv2.contourArea)

        # CVAT expects a flat list of points: [x1, y1, x2, y2, ..., xN, yN]
        # .squeeze() removes redundant dimensions, .ravel() flattens it.
        polygon_points = largest_contour.squeeze().ravel().tolist()

        # A valid polygon for CVAT needs at least 3 points (6 coordinates)
        if len(polygon_points) < 6:
            self.logger.warn(f"Contour found but resulted in too few points ({len(polygon_points)}) to form a valid polygon.") # Changed warning to warn
            return None

        return [float(p) for p in polygon_points] # Ensure all points are float

    def handle_detection(self, image: Image.Image, threshold: float | None = None):
        """
        Handle image input: run CNN detection, then optionally refine with SAM2 for masks.
        Accepts an optional threshold to override the class default.
        """
        final_detections = []

        # Determine the confidence threshold to use
        current_confidence_threshold = threshold if threshold is not None else self.CONFIDENCE_THRESHOLD
        if threshold is not None:
            self.logger.info(f"Using dynamic threshold for this request: {current_confidence_threshold}")
        else:
            self.logger.info(f"Using default class threshold: {current_confidence_threshold}")


        # 1. Run CNN Detection
        cnn_results = self._run_cnn_inference(image)
        self.logger.info(f"CNN produced {len(cnn_results)} raw detections.") # Log how many raw detections
        if cnn_results: # Log score statistics if there are any detections
            scores = [r['score'] for r in cnn_results]
            if scores: # Ensure scores list is not empty before min/max/avg
                self.logger.info(f"Raw detection scores: Min={min(scores):.4f}, Max={max(scores):.4f}, Avg={sum(scores)/len(scores):.4f}")
            else:
                self.logger.info("Raw detection scores: No scores available (empty list).")
        else:
            self.logger.info("Raw detection scores: No detections found.")


        # Prepare image for SAM2 if it's going to be used (once per image)
        # SAM2's default predictor expects a BGR NumPy array.
        image_np_bgr_for_sam = None
        can_run_sam2 = self.use_sam2_for_segmentation and CV2_AVAILABLE and self.sam2_predictor and cnn_results

        if can_run_sam2:
            try:
                image_np_rgb = np.array(image.convert("RGB"))
                image_np_bgr_for_sam = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
                self.sam2_predictor.set_image(image_np_bgr_for_sam)
                self.logger.info("Image successfully set for SAM2 predictor.")
            except Exception as e:
                self.logger.error(f"Error preparing image for SAM2: {e}. SAM2 will be skipped for this request.")
                can_run_sam2 = False # Disable SAM for this run if setup fails

        # 2. Process CNN results
        num_passed_threshold = 0 # Add a counter
        for cnn_det in cnn_results:
            score = cnn_det['score']

            if score < current_confidence_threshold: # Use the determined threshold
                continue
            
            num_passed_threshold += 1 # Increment if it passes the threshold
            self.logger.info(f"Detection with score {score:.4f} passed threshold. Attempting SAM2/fallback.") # Log when a detection passes

            label_id = cnn_det['label_id']
            box = cnn_det['box']  # [x1, y1, x2, y2]
            label_name = self.LABEL_MAP.get(label_id, f"unknown_id_{label_id}")

            # Attempt SAM2 refinement if conditions are met
            use_sam2_for_this_detection = can_run_sam2
            sam2_success = False

            if use_sam2_for_this_detection:
                try:
                    self.logger.info(f"Running SAM2 for box: {box} (label: {label_name}, score: {score:.2f})")
                    box_np = np.array(box).reshape(1, 4) # SAM2 expects (N, 4)

                    # multimask_output=False to get the single best mask
                    sam_masks, _, _ = self.sam2_predictor.predict(
                        box=box_np, # Changed from box_np to box
                        multimask_output=False
                    )

                    self.logger.info(f"Raw sam_masks type: {type(sam_masks)}")
                    if hasattr(sam_masks, 'shape'):
                        self.logger.info(f"Raw sam_masks shape: {sam_masks.shape}")
                    if hasattr(sam_masks, 'dtype'):
                        self.logger.info(f"Raw sam_masks dtype: {sam_masks.dtype}")
                    if isinstance(sam_masks, np.ndarray) and sam_masks.size > 0:
                        self.logger.info(f"Raw sam_masks min: {np.min(sam_masks)}, max: {np.max(sam_masks)}")
                        self.logger.info(f"Raw sam_masks unique values (first 10 if flat): {np.unique(sam_masks.ravel())[:10]}")
                    elif isinstance(sam_masks, list):
                        self.logger.info(f"Raw sam_masks is a list of length: {len(sam_masks)}")
                        if len(sam_masks) > 0:
                            first_element = sam_masks[0]
                            self.logger.info(f"Raw sam_masks first element type: {type(first_element)}")
                            if hasattr(first_element, 'shape'):
                                self.logger.info(f"Raw sam_masks first element shape: {first_element.shape}")


                    if sam_masks is not None and len(sam_masks) > 0:
                        # sam_masks shape is (N_prompts, H, W) when multimask_output=False
                        # For a single box_np prompt, it's (1, H, W)
                        # The predict method seems to return numpy arrays directly
                        mask_np = sam_masks[0] # Corrected indexing to get the full 2D mask

                        polygon_points = self._convert_mask_to_polygon(mask_np)

                        if polygon_points:
                            final_detections.append({
                                'type': 'mask',
                                'label': label_name,
                                'points': polygon_points,
                                'score': float(score) # Using CNN's score for the detection
                            })
                            self.logger.info(f"SAM2 mask converted to polygon for '{label_name}'.")
                            sam2_success = True
                        else:
                            self.logger.warn(f"SAM2 mask to polygon conversion failed for '{label_name}'. Falling back to bbox.") # Changed warning to warn
                    else:
                        self.logger.warn(f"SAM2 did not return a mask for '{label_name}'. Falling back to bbox.") # Changed warning to warn
                except Exception as e:
                    self.logger.error(f"Error during SAM2 refinement for box {box}: {str(e)}. Falling back to bbox.")

            # Fallback to bounding box if SAM2 was not used, failed, or didn't produce a valid polygon
            if not sam2_success:
                final_detections.append({
                    'type': 'rectangle',
                    'label': label_name,
                    'points': [float(c) for c in box], # [x1, y1, x2, y2]
                    'score': float(score)
                })
                if use_sam2_for_this_detection : # Log only if SAM2 was attempted
                     self.logger.info(f"Fell back to bounding box for '{label_name}' with score {score:.4f}.")


        self.logger.info(f"{num_passed_threshold} detections passed confidence threshold ({current_confidence_threshold}).") # Log total passed
        self.logger.info(f"Returning {len(final_detections)} final detections.")
        return final_detections
