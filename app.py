import json
import re
import threading
import av
import bs4
import cv2
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import numpy as np
import pypdf
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="Adaptive Vision Reader",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PWA & Standalone Mobile Meta Tags
pwa_meta_tags = """
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Adaptive Reader">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#0D1117">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <link rel="manifest" href="/manifest.json">
"""
st.markdown(pwa_meta_tags, unsafe_allow_html=True)


# Document Parsing & Web Scraping Helpers
def extract_text_from_pdf(uploaded_file):
  try:
    reader = pypdf.PdfReader(uploaded_file)
    extracted_pages = []
    for page in reader.pages:
      text = page.extract_text()
      if text:
        extracted_pages.append(text)
    return (
        "\n\n".join(extracted_pages)
        if extracted_pages
        else "No readable text found in PDF."
    )
  except Exception as e:
    return f"Error reading PDF: {str(e)}"


def extract_text_from_url(url):
  try:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=8)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text_content = [p.get_text().strip() for p in paragraphs if p.get_text()]
    if not text_content:
      return "Unable to extract main article body from this URL."
    return "\n\n".join(text_content)
  except Exception as e:
    return f"Failed to fetch content from URL: {str(e)}"


# Interactive Reader Component (Web Speech TTS + Sentence Highlighting + Dynamic Scaling)
def render_interactive_reader(
    text_content, font_size_px, is_high_contrast, speed=1.0
):
  if not text_content or text_content.startswith("Paste text"):
    text_content = "Paste text or select a content source above to begin."

  sentence_list = re.split(r"(?<=[.!?])\s+", text_content)
  sentences_data = []
  current_char_offset = 0

  for idx, sent in enumerate(sentence_list):
    if not sent.strip():
      continue
    sentences_data.append({
        "id": f"sent-{idx}",
        "text": sent,
        "start": current_char_offset,
        "end": current_char_offset + len(sent),
    })
    current_char_offset += len(sent) + 1

  sentences_json = json.dumps(sentences_data)
  raw_text_json = json.dumps(text_content)

  bg_color = "#0D1117" if is_high_contrast else "#FFFFFF"
  text_color = "#E6EDE3" if is_high_contrast else "#000000"
  highlight_bg = "#238636" if is_high_contrast else "#FFEB3B"
  highlight_text = "#FFFFFF" if is_high_contrast else "#000000"

  reader_html = f"""
    <style>
        .reader-container {{
            background-color: {bg_color};
            color: {text_color};
            font-size: {font_size_px}px;
            font-weight: 700;
            line-height: 1.7;
            padding: 24px;
            border-radius: 12px;
            font-family: system-ui, -apple-system, sans-serif;
            word-wrap: break-word;
            white-space: pre-wrap;
            max-height: 480px;
            overflow-y: auto;
            border: 1px solid #30363D;
        }}
        .sentence {{
            padding: 2px 4px;
            border-radius: 4px;
            transition: background-color 0.15s ease, color 0.15s ease;
        }}
        .active-sentence {{
            background-color: {highlight_bg} !important;
            color: {highlight_text} !important;
            box-shadow: 0 0 8px {highlight_bg};
        }}
        .controls-bar {{
            background-color: #161B22;
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 12px;
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
            font-family: system-ui, sans-serif;
        }}
        button {{
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            color: white;
        }}
    </style>

    <div class="controls-bar">
        <button onclick="startPlayback()" style="background-color: #238636;">▶ Play</button>
        <button onclick="pausePlayback()" style="background-color: #D29922;">⏸ Pause</button>
        <button onclick="resumePlayback()" style="background-color: #1F6FEB;">⏯ Resume</button>
        <button onclick="stopPlayback()" style="background-color: #DA3633;">⏹ Stop</button>
        
        <div style="margin-left: auto; color: #8B949E; font-size: 13px; display: flex; gap: 6px; align-items: center;">
            <label for="rate">Speed:</label>
            <input type="range" id="rate" min="0.5" max="2.0" value="{speed}" step="0.1" oninput="updateRateDisplay(this.value)" style="width: 80px;">
            <span id="rate-val" style="color: #E6EDE3; font-weight: bold;">{speed}x</span>
        </div>
    </div>

    <div id="text-view" class="reader-container"></div>

    <script>
        const sentences = {sentences_json};
        const fullText = {raw_text_json};
        const textView = document.getElementById('text-view');
        let utterance = null;
        let activeSpanId = null;

        function initializeTextDisplay() {{
            textView.innerHTML = '';
            sentences.forEach((s) => {{
                const span = document.createElement('span');
                span.id = s.id;
                span.className = 'sentence';
                span.innerText = s.text + ' ';
                textView.appendChild(span);
            }});
        }}

        initializeTextDisplay();

        function highlightSentenceAtCharIndex(charIndex) {{
            const matched = sentences.find(s => charIndex >= s.start && charIndex < s.end);
            
            if (matched && matched.id !== activeSpanId) {{
                if (activeSpanId) {{
                    const prevEl = document.getElementById(activeSpanId);
                    if (prevEl) prevEl.classList.remove('active-sentence');
                }}
                
                const newEl = document.getElementById(matched.id);
                if (newEl) {{
                    newEl.classList.add('active-sentence');
                    newEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
                activeSpanId = matched.id;
            }}
        }}

        function startPlayback() {{
            window.speechSynthesis.cancel();
            if (!fullText || fullText.trim() === "") return;

            utterance = new SpeechSynthesisUtterance(fullText);
            utterance.rate = parseFloat(document.getElementById('rate').value);

            utterance.addEventListener('boundary', (event) => {{
                if (event.name === 'word' || event.name === 'sentence') {{
                    highlightSentenceAtCharIndex(event.charIndex);
                }}
            }});

            utterance.onend = () => {{
                clearHighlights();
            }};

            window.speechSynthesis.speak(utterance);
        }}

        function pausePlayback() {{
            if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {{
                window.speechSynthesis.pause();
            }}
        }}

        function resumePlayback() {{
            if (window.speechSynthesis.paused) {{
                window.speechSynthesis.resume();
            }}
        }}

        function stopPlayback() {{
            window.speechSynthesis.cancel();
            clearHighlights();
        }}

        function clearHighlights() {{
            if (activeSpanId) {{
                const prevEl = document.getElementById(activeSpanId);
                if (prevEl) prevEl.classList.remove('active-sentence');
                activeSpanId = null;
            }}
        }}

        function updateRateDisplay(val) {{
            document.getElementById('rate-val').innerText = val + 'x';
            if (utterance && window.speechSynthesis.speaking) {{
                startPlayback();
            }}
        }}
    </script>
    """
  components.html(reader_html, height=580)


# Thread-Safe Asynchronous WebRTC Frame Processor
class FrameProcessor:

  def __init__(self):
    self.lock = threading.Lock()
    self.distance_cm = 40.0
    self.sharpen_amount = 1.2
    self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

  def apply_unsharp_mask(self, image, alpha):
    if alpha == 0:
      return image
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    return cv2.addWeighted(image, 1.0 + alpha, gaussian, -alpha, 0)

  def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    results = self.mp_face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
      landmarks = results.multi_face_landmarks[0].landmark
      left_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
      right_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])

      pixel_pd = np.linalg.norm(left_pupil - right_pupil)

      if pixel_pd > 0:
        calc_dist = (63.0 * 500.0) / (pixel_pd * 10.0)
        with self.lock:
          self.distance_cm = calc_dist

    with self.lock:
      alpha = self.sharpen_amount

    processed_img = self.apply_unsharp_mask(img, alpha)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# Session State Persistence
if "processor" not in st.session_state:
  st.session_state.processor = FrameProcessor()

processor = st.session_state.processor

# Header
st.title("👓 Adaptive Vision Reader")
st.caption(
    "Glasses-free reading via real-time camera tracking, dynamic typography,"
    " spatial sharpening, and synced TTS"
)

# Sidebar Calibration Controls
st.sidebar.header("Optical Calibration Settings")
near_point_cm = st.sidebar.slider("Near Point Focus Distance (cm)", 20, 60, 40)
base_font_pt = st.sidebar.slider("Base Font Size (pt)", 14, 32, 20)
sharpen_weight = st.sidebar.slider(
    "Spatial Sharpening Weight (α)", 0.0, 3.0, 1.2, 0.1
)
high_contrast = st.sidebar.checkbox("High Contrast Mode", value=True)

# Update sharpening amount inside frame processor thread
with processor.lock:
  processor.sharpen_amount = sharpen_weight

# ICE Server Configuration (STUN + Optional TURN secrets)
ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
if "turn" in st.secrets:
  ice_servers.append({
      "urls": [st.secrets["turn"]["url"]],
      "username": st.secrets["turn"]["username"],
      "credential": st.secrets["turn"]["credential"],
  })

rtc_config = RTCConfiguration({"iceServers": ice_servers})

# UI Layout Columns
col1, col2 = st.columns([1, 2])

with col1:
  st.subheader("Live Tracking Feed")
  webrtc_streamer(
      key="adaptive-reader",
      mode=WebRtcMode.SENDRECV,
      rtc_configuration=rtc_config,
      video_frame_callback=processor.recv,
      media_stream_constraints={"video": True, "audio": False},
  )

with col2:
  st.subheader("Adaptive Reader Display")

  input_mode = st.radio(
      "Content Source",
      ["Paste Text", "Upload PDF", "Web Link", "Sample Text"],
      horizontal=True,
  )

  active_text = ""

  if input_mode == "Paste Text":
    active_text = st.text_area(
        "Paste text below:",
        height=140,
        placeholder="Paste article or document text here...",
    )
    if not active_text:
      active_text = "Paste text above to start reading."

  elif input_mode == "Upload PDF":
    pdf_file = st.file_uploader("Upload a document", type=["pdf"])
    if pdf_file is not None:
      active_text = extract_text_from_pdf(pdf_file)
    else:
      active_text = "Upload a PDF document to parse text."

  elif input_mode == "Web Link":
    target_url = st.text_input("Enter Web URL:", placeholder="https://...")
    if target_url:
      with st.spinner("Extracting article text..."):
        active_text = extract_text_from_url(target_url)
    else:
      active_text = "Paste a web article link above."

  elif input_mode == "Sample Text":
    active_text = (
        "High spatial frequency components (sharp edges) fade first when eyes"
        " lose focal accommodation. By applying an unsharp mask convolution"
        " kernel in real time, character borders receive a localized contrast"
        " boost before light hits the eye. Combined with WebRTC pupillary"
        " tracking, font dimensions scale dynamically to keep text completely"
        " legible without requiring readers."
    )

  with processor.lock:
    current_distance = processor.distance_cm

  st.metric("Continuous Measured Distance", f"{current_distance:.1f} cm")

  if current_distance < near_point_cm:
    scale_factor = near_point_cm / max(current_distance, 15.0)
  else:
    scale_factor = 1.0

  dynamic_font_px = int(base_font_pt * scale_factor * 1.33)

  render_interactive_reader(
      text_content=active_text,
      font_size_px=dynamic_font_px,
      is_high_contrast=high_contrast,
  )
