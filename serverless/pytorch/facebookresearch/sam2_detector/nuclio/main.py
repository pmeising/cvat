# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
# We'll rename ModelHandler or create a new one that handles detection
# Let's call it DetectorHandler for clarity
from model_handler import DetectorHandler

def init_context(context):
    # Initialize the DetectorHandler which loads both models (CNN and SAM2)
    # Pass context.logger to the handler for logging within the class
    try:
        model = DetectorHandler(context.logger)
        context.user_data.model = model
        context.logger.info("Init context...100%")
    except Exception as e:
        context.logger.error(f"Error during init_context: {str(e)}")
        # Depending on Nuclio configuration, initialization failure might prevent the function from starting
        raise # Re-raise the exception to indicate initialization failure

def handler(context, event):
    try:
        context.logger.info("Detector handler called")
        data = event.body

        # Detector input only contains the image
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB") # Ensure image is in RGB format

        # --- Removed interactor-specific input handling ---
        # No longer expect data["pos_points"] or data["neg_points"]
        # --------------------------------------------------

        # Call the new detection method on the model handler
        # This method will return a list of detected objects
        detections = context.user_data.model.handle_detection(image)

        # The response should be a JSON array of detection objects
        # Each object in the array should contain type, label, points, score
        # Example: [{'type': 'mask', 'label': 'person', 'points': [...], 'score': 0.95}, ...]

        return context.Response(
            body=json.dumps(detections), # Return the list of detections directly
            headers={},
            content_type='application/json',
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"Error in detector handler: {str(e)}")
        return context.Response(
            body=json.dumps({'error': str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )