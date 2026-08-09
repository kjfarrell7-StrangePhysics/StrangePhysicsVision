import threading
import cv2
import numpy as np
import streamlit as st
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Initialize MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh


class FrameProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.distance_cm = 40.0
        self.sharpen_amount = 1.2
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw facial mesh keypoints on frame
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)

                # Estimate distance using inter-ocular distance (landmarks 33 and 263)
                p1 = face_landmarks.landmark[33]
                p2 = face_landmarks.landmark[263]
                dist_px = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) * w
                if dist_px > 0:
                    calculated_dist = round(float((6.3 * 500) / dist_px), 1)
                    with self.lock:
                        self.distance_cm = calculated_dist

        with self.lock:
            sharpen = self.sharpen_amount

        # Apply spatial sharpening kernel if enabled
        if sharpen > 1.0:
            kernel = np.array(
                [[0, -1, 0], [-1, 4 + sharpen, -1], [0, -1, 0]]
            )
            image = cv2.filter2D(image, -1, kernel)

        return image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = self.process_frame(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# Streamlit Page Configuration
st.set_page_config(
    page_title="StrangePhysics Vision",
    page_icon="👁️",
    layout="wide",
)

st.title("StrangePhysics Vision System 👁️⚡")
st.markdown(
    "Real-time webcam video stream with facial landmark tracking, distance estimation, and dynamic image processing filters."
)

# Sidebar Controls & Live Telemetry
st.sidebar.header("Vision Controls")

if "processor" not in st.session_state:
    st.session_state.processor = FrameProcessor()

processor = st.session_state.processor

sharpen_val = st.sidebar.slider("Sharpen Amount", 1.0, 5.0, 1.2, 0.1)
with processor.lock:
    processor.sharpen_amount = sharpen_val

st.sidebar.markdown("---")
st.sidebar.metric("Estimated Face Distance", f"{processor.distance_cm} cm")

# WebRTC Connection Settings
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.subheader("Live Feed")
webrtc_streamer(
    key="strangephysics-vision",
    video_processor_factory=lambda: processor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)
