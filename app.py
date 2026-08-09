import threading
import bs4
import cv2
import numpy as np
import pypdf
import requests
import streamlit as st
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Safely import FaceMesh across different MediaPipe release structures
try:
    FaceMesh = mp.solutions.face_mesh.FaceMesh
except AttributeError:
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        FaceMesh = mp_face_mesh.FaceMesh
    except (ImportError, AttributeError):
        from mediapipe.solutions import face_mesh
        FaceMesh = face_mesh.FaceMesh


class FrameProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.distance_cm = 40.0
        self.sharpen_amount = 1.2
        self.mp_face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw facial mesh landmarks
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)

                # Estimate face distance using eye landmarks (33 and 263)
                p1 = face_landmarks.landmark[33]
                p2 = face_landmarks.landmark[263]
                dist_px = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) * w
                if dist_px > 0:
                    calculated_dist = round(float((6.3 * 500) / dist_px), 1)
                    with self.lock:
                        self.distance_cm = calculated_dist

        # Apply sharpening filter if adjusted above 1.0
        with self.lock:
            sharpen = self.sharpen_amount

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


# Page Layout & Configuration
st.set_page_config(
    page_title="StrangePhysics Vision",
    page_icon="👁️",
    layout="wide",
)

st.title("StrangePhysics Vision System 👁️⚡")
st.markdown(
    "Real-time webcam video stream with facial landmark tracking, distance estimation, and image processing controls."
)

# Sidebar Controls
st.sidebar.header("Vision Controls")

# Session State Persistence
if "processor" not in st.session_state:
    st.session_state.processor = FrameProcessor()

processor = st.session_state.processor

sharpen_val = st.sidebar.slider("Sharpen Amount", 1.0, 5.0, 1.2, 0.1)
with processor.lock:
    processor.sharpen_amount = sharpen_val

st.sidebar.markdown("---")
st.sidebar.metric("Estimated Face Distance", f"{processor.distance_cm} cm")

# WebRTC Streamer Setup
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
