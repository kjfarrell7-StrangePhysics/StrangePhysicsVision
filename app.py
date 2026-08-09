import threading
import cv2
import numpy as np
import streamlit as st
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

mp_face_mesh = mp.solutions.face_mesh


class FrameProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.distance_cm = 40.0
        self.sharpen_amount = 1.2
        self.zoom_factor = 1.0
        self.high_contrast = False
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
                # Inter-ocular distance calculation
                p1 = face_landmarks.landmark[33]
                p2 = face_landmarks.landmark[263]
                dist_px = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) * w
                if dist_px > 0:
                    calculated_dist = round(float((6.3 * 500) / dist_px), 1)
                    with self.lock:
                        self.distance_cm = calculated_dist

        with self.lock:
            sharpen = self.sharpen_amount
            zoom = self.zoom_factor
            contrast = self.high_contrast

        # 1. Digital Zoom / Cropping
        if zoom > 1.0:
            crop_h, crop_w = int(h / zoom), int(w / zoom)
            start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
            cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
            image = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Sharpening Filter
        if sharpen > 1.0:
            kernel = np.array([[0, -1, 0], [-1, 4 + sharpen, -1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)

        # 3. High Contrast Mode
        if contrast:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

        return image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = self.process_frame(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# Page Setup
st.set_page_config(page_title="Adaptive Vision Reader", page_icon="👁️", layout="wide")

st.title("Adaptive Vision Reader 👁️📖")
st.markdown("Dynamic screen scaling & distance-based auto-magnification.")

# Sidebar Adjustments
st.sidebar.header("Vision & Magnification Settings")

if "processor" not in st.session_state:
    st.session_state.processor = FrameProcessor()

processor = st.session_state.processor

sharpen_val = st.sidebar.slider("Sharpening", 1.0, 5.0, 1.5, 0.1)
zoom_val = st.sidebar.slider("Camera Digital Zoom", 1.0, 3.0, 1.2, 0.1)
high_contrast_val = st.sidebar.checkbox("High Contrast Mode", False)

with processor.lock:
    processor.sharpen_amount = sharpen_val
    processor.zoom_factor = zoom_val
    processor.high_contrast = high_contrast_val

current_dist = processor.distance_cm
st.sidebar.markdown("---")
st.sidebar.metric("Eye-to-Screen Distance", f"{current_dist} cm")

# Adaptive Font Sizing Logic
# Calculates dynamic font size: farther distance = larger text size
calculated_font_size = max(18, int((current_dist / 30.0) * 24))

# UI Tabs: Reader Mode vs Video Feed
tab1, tab2 = st.tabs(["📖 Reading Canvas", "📷 Live Camera Stream"])

with tab1:
    st.markdown(f"### Adaptive Reader Output (Font Size: {calculated_font_size}px)")
    
    user_text = st.text_area(
        "Paste or type text to magnify:",
        value="Optics and wave mechanics demonstrate how spatial frequencies change with distance. Moving the display to arm's length reduces perceived retinal angle, requiring dynamic scaling for sharp focus without auxiliary lenses.",
        height=100,
    )
    
    # Styled output container with dynamically injected font-size
    st.markdown(
        f"""
        <div style="
            font-size: {calculated_font_size}px; 
            line-height: 1.6; 
            padding: 20px; 
            background-color: #1e1e1e; 
            color: #ffffff; 
            border-radius: 10px; 
            border: 2px solid #4CAF50;">
            {user_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(
        key="strangephysics-vision",
        video_processor_factory=lambda: processor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )
