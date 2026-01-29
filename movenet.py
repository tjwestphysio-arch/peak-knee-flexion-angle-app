import numpy as np
import cv2
import tensorflow as tf

# COCO keypoint indices for MoveNet single-pose
# 11: left_hip, 13: left_knee, 15: left_ankle
# 12: right_hip, 14: right_knee, 16: right_ankle
LEFT_HIP = 11
LEFT_KNEE = 13
LEFT_ANKLE = 15
RIGHT_HIP = 12
RIGHT_KNEE = 14
RIGHT_ANKLE = 16


class MoveNetLightning:
    def __init__(self, model_path: str = "movenet_singlepose_lightning.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.input_height = input_details[0]["shape"][1]
        self.input_width = input_details[0]["shape"][2]
        self.output_index = output_details[0]["index"]

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        # Convert BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_tensor = resized.astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor

    def infer_keypoints(self, image_bgr: np.ndarray) -> np.ndarray:
        """Returns keypoints with shape (17, 3): [y, x, score]."""
        input_tensor = self._preprocess(image_bgr)
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_index)
        # keypoints_with_scores: [1, 1, 17, 3]
        keypoints = keypoints_with_scores[0, 0, :, :]  # (17, 3)
        return keypoints

    @staticmethod
    def _angle_between_points(a, b, c):
        """
        Compute angle at point b given three points a, b, c in (y, x) format.
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])

        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return None

        cos_angle = dot_product / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def compute_knee_angles(self, keypoints: np.ndarray, score_threshold: float = 0.3):
        """
        Compute left and right knee flexion angles (in degrees) if landmarks are confident enough.
        Returns a dict with 'left_knee' and 'right_knee' (may be None).
        """
        def valid(idx):
            return keypoints[idx, 2] >= score_threshold

        left_angle = None
        right_angle = None

        # Left knee
        if valid(LEFT_HIP) and valid(LEFT_KNEE) and valid(LEFT_ANKLE):
            a = keypoints[LEFT_HIP, :2]
            b = keypoints[LEFT_KNEE, :2]
            c = keypoints[LEFT_ANKLE, :2]
            left_angle = self._angle_between_points(a, b, c)

        # Right knee
        if valid(RIGHT_HIP) and valid(RIGHT_KNEE) and valid(RIGHT_ANKLE):
            a = keypoints[RIGHT_HIP, :2]
            b = keypoints[RIGHT_KNEE, :2]
            c = keypoints[RIGHT_ANKLE, :2]
            right_angle = self._angle_between_points(a, b, c)

        return {
            "left_knee": left_angle,
            "right_knee": right_angle,
        }

    def draw_keypoints_and_angles(self, image_bgr: np.ndarray, keypoints: np.ndarray, angles: dict):
        """
        Draw keypoints and knee angles on the image.
        """
        h, w, _ = image_bgr.shape
        output = image_bgr.copy()

        # Draw keypoints
        for i in range(keypoints.shape[0]):
            y, x, score = keypoints[i]
            if score < 0.3:
                continue
            cx, cy = int(x * w), int(y * h)
            cv2.circle(output, (cx, cy), 3, (0, 255, 0), -1)

        # Draw angles near knees
        for side, idx in [("left_knee", LEFT_KNEE), ("right_knee", RIGHT_KNEE)]:
            angle = angles.get(side)
            if angle is None:
                continue
            y, x, score = keypoints[idx]
            if score < 0.3:
                continue
            cx, cy = int(x * w), int(y * h)
            cv2.putText(
                output,
                f"{side.replace('_', ' ').title()}: {angle:.1f} deg",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output
