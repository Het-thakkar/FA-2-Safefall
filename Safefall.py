import streamlit as st
import cv2, os, io, wave, base64, threading, time
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# streamlit-webrtc powers the real continuous live camera feed + per-frame inference.
# (pip install streamlit-webrtc streamlit-autorefresh -- see requirements.txt)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="AI Elderly Fall Detection", layout="wide")

# ---------------------------------------------------------------
# Model + pose landmarker loading (unchanged from original)
# ---------------------------------------------------------------
@st.cache_resource
def load_rf_model():
    model_path = 'fall_detection_rf.pkl'
    if not os.path.exists(model_path):
        st.error("Missing model: 'fall_detection_rf.pkl'. Run `train_model.py` first.")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
        st.stop()


@st.cache_resource
def load_landmarker():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        st.error("Missing 'pose_landmarker_lite.task'")
        st.stop()
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            min_pose_detection_confidence=0.5
        )
    )

rf_model = load_rf_model()
landmarker = load_landmarker()

# 5 activity classes, matching train_model.py / extract_features.py
CLASS_LABELS = {
    0: "⚠️ FALL DETECTED",
    1: "✅ Normal Activity",
}
FALL_CLASS_IDX = 0
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def calc_angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180.0 else ang


def extract_invariant_features(landmarks):
    nose = landmarks[0]; l_sh = landmarks[11]; r_sh = landmarks[12]
    l_el = landmarks[13]; r_el = landmarks[14]
    l_wr = landmarks[15]; r_wr = landmarks[16]
    l_hip = landmarks[23]; r_hip = landmarks[24]
    l_knee = landmarks[25]; r_knee = landmarks[26]
    l_ankle = landmarks[27]; r_ankle = landmarks[28]
    l_foot = landmarks[31]; r_foot = landmarks[32]

    head_tilt = calc_angle(l_sh, nose, r_sh)
    l_sh_angle = calc_angle(r_sh, l_sh, nose); r_sh_angle = calc_angle(l_sh, r_sh, nose)
    l_elbow = calc_angle(l_sh, l_el, l_wr); r_elbow = calc_angle(r_sh, r_el, r_wr)

    if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
        l_hip_angle, r_hip_angle, l_knee_angle, r_knee_angle, l_ankle_angle, r_ankle_angle = \
            calc_angle(l_sh, l_hip, l_knee), calc_angle(r_sh, r_hip, r_knee), \
            calc_angle(l_hip, l_knee, l_ankle), calc_angle(r_hip, r_knee, r_ankle), \
            calc_angle(l_knee, l_ankle, l_foot), calc_angle(r_knee, r_ankle, r_foot)
    else:
        l_hip_angle = r_hip_angle = l_knee_angle = r_knee_angle = l_ankle_angle = r_ankle_angle = 0

    sh_vec = np.array([r_sh.x - l_sh.x, r_sh.y - l_sh.y])
    sh_line_angle = np.arctan2(sh_vec[1], sh_vec[0]) * 180.0 / np.pi

    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2])
        mid_hip = np.array([(l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2])
        torso_angle = np.arctan2((mid_hip - mid_sh)[1], (mid_hip - mid_sh)[0]) * 180.0 / np.pi
        hip_line_angle = np.arctan2((r_hip.x - l_hip.x), (r_hip.y - l_hip.y)) * 180.0 / np.pi
    else:
        torso_angle, hip_line_angle = 0, 0

    shoulder_width = np.linalg.norm(sh_vec)
    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2])
        mid_hip = np.array([(l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2])
        torso_len = np.linalg.norm(mid_sh - mid_hip)
        avg_leg_len = (np.linalg.norm(np.array([l_hip.x, l_hip.y]) - np.array([l_ankle.x, l_ankle.y])) +
                       np.linalg.norm(np.array([r_hip.x, r_hip.y]) - np.array([r_ankle.x, r_ankle.y]))) / 2
        leg_torso_ratio = avg_leg_len / torso_len if torso_len > 0.001 else 0
    else:
        leg_torso_ratio = 0

    return np.array([head_tilt, l_sh_angle, r_sh_angle, l_elbow, r_elbow, l_hip_angle, r_hip_angle,
                      l_knee_angle, r_knee_angle, l_ankle_angle, r_ankle_angle, torso_angle,
                      sh_line_angle, hip_line_angle, shoulder_width, leg_torso_ratio]).reshape(1, -1)


def draw_landmarks_on_image(image, result):
    if not result.pose_landmarks:
        return image
    h, w, _ = image.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.pose_landmarks[0]]
    for conn in POSE_CONNECTIONS:
        if conn[0] < len(pts) and conn[1] < len(pts):
            cv2.line(image, pts[conn[0]], pts[conn[1]], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(image, pt, 3, (0, 0, 255), -1)
    return image


def run_prediction(img_rgb):
    """Shared inference used by both the upload path and the live video path."""
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    annotated = draw_landmarks_on_image(img_rgb.copy(), result)
    if not result.pose_landmarks:
        return annotated, None, None
    feats = extract_invariant_features(result.pose_landmarks[0])
    # `rf_model` is a saved pipeline (scaler + classifier). Call predict_proba on raw features.
    pred_probs = rf_model.predict_proba(feats)
    pred_idx = int(np.argmax(pred_probs[0]))
    conf = float(np.max(pred_probs[0]))
    if pred_idx == FALL_CLASS_IDX:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 255), 8)
        cv2.putText(annotated, "FALL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    else:
        cv2.putText(annotated, CLASS_LABELS[pred_idx], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return annotated, pred_idx, conf


# ---------------------------------------------------------------
# Alarm sound: generate a short beep on the fly (no audio file needed)
# ---------------------------------------------------------------
def generate_beep_base64(freq=1000, duration_ms=400, volume=0.5, sample_rate=44100):
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    tone = np.sin(freq * t * 2 * np.pi) * volume
    audio = (tone * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


BEEP_B64 = generate_beep_base64()


def play_alarm():
    """Injects an auto-playing <audio> tag. Each call plays the beep once."""
    st.markdown(
        f"""<audio autoplay="true">
            <source src="data:audio/wav;base64,{BEEP_B64}" type="audio/wav">
            </audio>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# Live video processor — runs continuously on every incoming webcam frame
# ---------------------------------------------------------------
class FallVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_pred_idx = None
        self.latest_conf = 0.0
        self.last_frame_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Throttle inference slightly to keep the video smooth (process ~every other frame)
        now = time.time()
        run_inference = (now - self.last_frame_time) > 0.15
        if run_inference:
            self.last_frame_time = now
            annotated_rgb, pred_idx, conf = run_prediction(img_rgb)
            with self.lock:
                self.latest_pred_idx = pred_idx
                self.latest_conf = conf if conf is not None else 0.0
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        else:
            annotated_bgr = img

        return frame.from_ndarray(annotated_bgr, format="bgr24")

    def get_latest(self):
        with self.lock:
            return self.latest_pred_idx, self.latest_conf


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🏥 AI Elderly Fall Detection (Random Forest)")

for key, default in [('fall', 0), ('normal', 0), ('was_falling', False)]:
    if key not in st.session_state:
        st.session_state[key] = default

input_mode = st.radio(
    "Input source",
    ["📁 Upload Image", "🔴 Live Real-Time Tracking"],
    horizontal=True,
)

col1, col2 = st.columns([2, 1])

# ---------------- Upload Image mode (unchanged behaviour) ----------------
if input_mode == "📁 Upload Image":
    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = np.array(Image.open(uploaded).convert("RGB"))
        annotated_img, pred_idx, conf = run_prediction(img)

        if pred_idx is None:
            col1.warning("No pose detected.")
        elif pred_idx == FALL_CLASS_IDX:
            st.session_state.fall += 1
            col2.error("🚨 EMERGENCY ALERT: FALL DETECTED!")
            play_alarm()
            col1.image(annotated_img, use_column_width=True)
        else:
            st.session_state.normal += 1
            col2.success(f"{CLASS_LABELS[pred_idx]} ({conf:.2f})")
            col1.image(annotated_img, use_column_width=True)

# ---------------- Live Real-Time Tracking mode ----------------
else:
    col1.caption(
        "Click **Start** below, allow camera access, and stay in frame. "
        "The system analyses your pose continuously — an alarm sounds automatically "
        "if a fall is detected and sustained for a moment (avoids false alarms from a single bad frame)."
    )

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = col1.empty()
    with col1:
        ctx = webrtc_streamer(
            key="fall-detection-live",
            video_processor_factory=FallVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )

    status_box = col2.empty()
    metrics_box = col2.container()

    if ctx.state.playing:
        # Rerun this script every second so we can poll the video processor's
        # latest result and update the alert / sound without touching the video loop.
        st_autorefresh(interval=1000, key="live_refresh")

        if ctx.video_processor:
            pred_idx, conf = ctx.video_processor.get_latest()

            if pred_idx == FALL_CLASS_IDX:
                status_box.error(f"🚨 EMERGENCY ALERT: FALL DETECTED! (confidence {conf:.2f})")
                play_alarm()
                if not st.session_state.was_falling:
                    st.session_state.fall += 1
                    st.session_state.was_falling = True
            elif pred_idx is not None:
                status_box.success(f"{CLASS_LABELS[pred_idx]} (confidence {conf:.2f})")
                if st.session_state.was_falling:
                    st.session_state.normal += 1
                st.session_state.was_falling = False
            else:
                status_box.info("Waiting for a person to appear in frame...")
    else:
        status_box.info("Click **Start** above to begin live tracking.")

# ---------------- Sidebar-style metrics + evaluation plots (always visible) ----------------
with col2:
    total = st.session_state.fall + st.session_state.normal
    st.metric("Total", total)
    st.metric("Falls", st.session_state.fall)
    st.metric("Normals", st.session_state.normal)
    if os.path.exists("evaluation_plots"):
        try:
            st.image("evaluation_plots/confusion_matrix.png", use_column_width=True)
            st.image("evaluation_plots/feature_importance.png", use_column_width=True)
        except:
            pass
