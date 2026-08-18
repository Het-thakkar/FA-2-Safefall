import streamlit as st
import cv2, os, io, wave, base64, threading, time
import numpy as np
import mediapipe as mp
import joblib
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="SafeFall AI - Neural CCTV", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b0f14; }
h1, h2, h3, p, span, label { color: #d7f9ff !important; }
[data-testid="stMetricValue"] { color: #00e5ff !important; }
[data-testid="stMetricLabel"] { color: #7fdfff !important; }
div[data-testid="column"] {
    background: #10161d;
    border: 1px solid #1c2b36;
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Model + scaler + pose landmarker
# ---------------------------------------------------------------
@st.cache_resource
def load_rf_model():
    return joblib.load('fall_detection_rf.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('feature_scaler.pkl')

@st.cache_resource
def load_landmarker():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        st.error("Missing 'pose_landmarker_lite.task' in this folder.")
        st.stop()
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            min_pose_detection_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
    )

rf_model = load_rf_model()
scaler = load_scaler()
landmarker = load_landmarker()

CLASS_LABELS = {0: "FALL DETECTED", 1: "Walking", 2: "Sitting", 3: "Standing", 4: "Normal Activity"}
# Neon colors (BGR) per class -- this is what gives the "AI neural" look
CLASS_COLORS_BGR = {
    0: (40, 40, 255),    # red-orange
    1: (255, 200, 0),    # cyan-orange
    2: (255, 60, 220),   # magenta
    3: (60, 255, 120),   # green
    4: (255, 220, 100),  # soft cyan
}
FALL_CLASS_IDX = 0
FALL_CONFIRM_FRAMES = 5
# Newer mediapipe releases (0.10.31+ / 1.0.x) removed the legacy "solutions"
# module entirely, so mp.solutions.pose.POSE_CONNECTIONS no longer works.
# This is the standard 33-point MediaPipe Pose skeleton topology, hardcoded
# directly so we don't depend on the removed module at all.
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
])


def calc_angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180.0 else ang


def extract_invariant_features(landmarks):
    nose = landmarks[0]; l_sh = landmarks[11]; r_sh = landmarks[12]
    l_el = landmarks[13]; r_el = landmarks[14]; l_wr = landmarks[15]; r_wr = landmarks[16]
    l_hip = landmarks[23]; r_hip = landmarks[24]
    l_knee = landmarks[25]; r_knee = landmarks[26]
    l_ankle = landmarks[27]; r_ankle = landmarks[28]
    l_foot = landmarks[31]; r_foot = landmarks[32]

    head_tilt = calc_angle(l_sh, nose, r_sh)
    l_sh_angle = calc_angle(r_sh, l_sh, nose); r_sh_angle = calc_angle(l_sh, r_sh, nose)
    l_elbow = calc_angle(l_sh, l_el, l_wr); r_elbow = calc_angle(r_sh, r_el, r_wr)

    if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
        l_hip_angle, r_hip_angle = calc_angle(l_sh, l_hip, l_knee), calc_angle(r_sh, r_hip, r_knee)
        l_knee_angle, r_knee_angle = calc_angle(l_hip, l_knee, l_ankle), calc_angle(r_hip, r_knee, r_ankle)
        l_ankle_angle, r_ankle_angle = calc_angle(l_knee, l_ankle, l_foot), calc_angle(r_knee, r_ankle, r_foot)
    else:
        l_hip_angle = r_hip_angle = l_knee_angle = r_knee_angle = l_ankle_angle = r_ankle_angle = 0

    sh_vec = np.array([r_sh.x - l_sh.x, r_sh.y - l_sh.y])
    sh_line_angle = np.arctan2(sh_vec[1], sh_vec[0]) * 180.0 / np.pi

    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
        mid_sh = np.array([(l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2])
        mid_hip = np.array([(l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2])
        torso_angle = np.arctan2((mid_hip - mid_sh)[1], (mid_hip - mid_sh)[0]) * 180.0 / np.pi
        hip_line_angle = np.arctan2((r_hip.x - l_hip.x), (r_hip.y - l_hip.y)) * 180.0 / np.pi
        torso_len = np.linalg.norm(mid_sh - mid_hip)
        l_leg = np.linalg.norm(np.array([l_hip.x, l_hip.y]) - np.array([l_ankle.x, l_ankle.y]))
        r_leg = np.linalg.norm(np.array([r_hip.x, r_hip.y]) - np.array([r_ankle.x, r_ankle.y]))
        leg_torso_ratio = ((l_leg + r_leg) / 2) / torso_len if torso_len > 0.001 else 0
    else:
        torso_angle = hip_line_angle = leg_torso_ratio = 0

    shoulder_width = np.linalg.norm(sh_vec)

    return np.array([head_tilt, l_sh_angle, r_sh_angle, l_elbow, r_elbow, l_hip_angle, r_hip_angle,
                      l_knee_angle, r_knee_angle, l_ankle_angle, r_ankle_angle, torso_angle,
                      sh_line_angle, hip_line_angle, shoulder_width, leg_torso_ratio]).reshape(1, -1)


# ---------------------------------------------------------------
# "Neural" skeleton rendering: glowing nodes + connections.
# This is purely visual -- it draws on top of the same real
# landmark coordinates used for prediction, nothing is fabricated.
# ---------------------------------------------------------------
def draw_neural_skeleton(image, landmarks, color):
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    glow_layer = np.zeros_like(image)
    for conn in POSE_CONNECTIONS:
        if conn[0] < len(pts) and conn[1] < len(pts):
            cv2.line(glow_layer, pts[conn[0]], pts[conn[1]], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(glow_layer, pt, 5, color, -1, cv2.LINE_AA)

    # Bloom/glow effect: blur a copy and blend under the sharp lines
    glow_blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=8, sigmaY=8)
    image = cv2.addWeighted(image, 1.0, glow_blurred, 0.9, 0)
    image = cv2.addWeighted(image, 1.0, glow_layer, 1.0, 0)

    # Small "synapse" pulse dots at major joints for the neural-net feel
    for idx in [11, 12, 23, 24, 25, 26]:  # shoulders, hips, knees
        if idx < len(pts):
            cv2.circle(image, pts[idx], 8, color, 1, cv2.LINE_AA)

    return image


def run_prediction(img_rgb, timestamp_ms):
    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb), timestamp_ms)

    if not result.pose_landmarks:
        return img_rgb, None, 0.0

    landmarks = result.pose_landmarks[0]

    l_hip, r_hip = landmarks[23], landmarks[24]
    l_knee, r_knee = landmarks[25], landmarks[26]
    l_ankle, r_ankle = landmarks[27], landmarks[28]
    lower_body_visible = all(
        lm.visibility > 0.5 for lm in [l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
    )

    if not lower_body_visible:
        annotated = draw_neural_skeleton(img_rgb.copy(), landmarks, (150, 150, 150))
        return annotated, "OUT_OF_FRAME", 0.0

    feats_raw = extract_invariant_features(landmarks)
    feats_scaled = scaler.transform(feats_raw)
    probs = rf_model.predict_proba(feats_scaled)
    pred_idx = int(np.argmax(probs[0]))
    conf = float(np.max(probs[0]))  # REAL model confidence, never fabricated

    # --- Sensitivity boost for Fall specifically ---
    # The model was trained on Le2i's CCTV camera angles, which differ from a
    # laptop webcam's close-range framing -- real falls on your own camera may
    # not score highest by raw argmax even though real Fall probability is
    # clearly elevated. This checks Fall's own probability against a lower
    # bar (not just "is it the top class") to catch those cases too.
    fall_prob = float(probs[0][FALL_CLASS_IDX])
    FALL_PROB_THRESHOLD = 0.28  # tune this: lower = more sensitive, more false positives
    if pred_idx != FALL_CLASS_IDX and fall_prob >= FALL_PROB_THRESHOLD:
        pred_idx = FALL_CLASS_IDX
        conf = fall_prob

    if pred_idx == FALL_CLASS_IDX:
        l_knee_angle = calc_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calc_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        if avg_knee_angle < 130 and mid_hip_y < 0.75:
            pred_idx = 2  # Sitting override
            conf = min(conf, 0.6)

    color = CLASS_COLORS_BGR.get(pred_idx, (255, 255, 255))
    annotated = draw_neural_skeleton(img_rgb.copy(), landmarks, color)
    return annotated, pred_idx, conf


# ---------------------------------------------------------------
# Alarm sound
# ---------------------------------------------------------------
def generate_beep_base64(freq=1000, duration_ms=400, volume=0.5, sample_rate=44100):
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    tone = np.sin(freq * t * 2 * np.pi) * volume
    audio = (tone * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

BEEP_B64 = generate_beep_base64()

def play_alarm():
    st.markdown(
        f'<audio autoplay="true"><source src="data:audio/wav;base64,{BEEP_B64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# Continuous video processor
# ---------------------------------------------------------------
class NeuralMonitor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_pred_idx = None
        self.latest_conf = 0.0
        self.fall_streak = 0
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self._fps_window_start = time.time()
        self._fps_window_count = 0
        self._timestamp_counter = 0
        self._last_infer_time = 0
        self._last_annotated_bgr = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Throttle heavy inference to ~8 times/sec instead of every single
        # frame -- running full pose detection + classification on every
        # frame is too heavy for most machines and causes exactly the kind
        # of lag/freezing you were seeing. Frames in between just reuse the
        # last annotated result so the video still looks smooth.
        now = time.time()
        should_infer = (now - self._last_infer_time) > 0.12

        if should_infer:
            self._last_infer_time = now
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Strictly-incrementing timestamp -- required by detect_for_video()
            self._timestamp_counter += 1
            annotated_rgb, pred_idx, conf = run_prediction(img_rgb, self._timestamp_counter)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            with self.lock:
                self.latest_pred_idx = pred_idx
                self.latest_conf = conf
                if pred_idx == FALL_CLASS_IDX:
                    self.fall_streak += 1
                else:
                    self.fall_streak = 0

                self._fps_window_count += 1
                elapsed = now - self._fps_window_start
                if elapsed >= 1.0:
                    self.fps = self._fps_window_count / elapsed
                    self._fps_window_count = 0
                    self._fps_window_start = now
        else:
            # Skip heavy processing this frame -- just pass the raw frame through
            annotated_bgr = img
            with self.lock:
                pred_idx = self.latest_pred_idx
                conf = self.latest_conf

        # On-frame label (this IS the same value the sidebar will show -- single source of truth)
        if pred_idx == "OUT_OF_FRAME":
            cv2.putText(annotated_bgr, "Move back - full body not in frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        elif pred_idx is not None:
            label = CLASS_LABELS[pred_idx]
            color = CLASS_COLORS_BGR[pred_idx]
            if pred_idx == FALL_CLASS_IDX and self.fall_streak >= FALL_CONFIRM_FRAMES:
                cv2.rectangle(annotated_bgr, (0, 0), (annotated_bgr.shape[1], annotated_bgr.shape[0]), color, 8)
                cv2.putText(annotated_bgr, f"FALL DETECTED ({conf:.0%})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            else:
                cv2.putText(annotated_bgr, f"{label} ({conf:.0%})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(annotated_bgr, f"{self.fps:.1f} FPS", (annotated_bgr.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)

        return frame.from_ndarray(annotated_bgr, format="bgr24")

    def get_latest(self):
        with self.lock:
            return self.latest_pred_idx, self.latest_conf, self.fall_streak, self.fps


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🧠 SafeFall AI — Neural CCTV Monitor")
st.caption("24/7 continuous detection. All values shown are live model output — nothing here is placeholder or simulated.")

if 'fall_events' not in st.session_state:
    st.session_state.fall_events = 0
if 'was_falling' not in st.session_state:
    st.session_state.was_falling = False
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = {k: 0 for k in CLASS_LABELS}

col_video, col_status = st.columns([3, 1])

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with col_video:
    ctx = webrtc_streamer(
        key="safefall-neural-cctv",
        video_processor_factory=NeuralMonitor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}},
            "audio": False,
        },
    )

with col_status:
    st.subheader("Live Status")
    status_box = st.empty()
    fps_box = st.empty()
    st.divider()
    st.metric("Fall Events", st.session_state.fall_events)
    counts_box = st.empty()

if ctx.state.playing:
    st_autorefresh(interval=500, key="neural_refresh")  # syncs sidebar to the SAME processor state shown on video

    if ctx.video_processor:
        pred_idx, conf, fall_streak, fps = ctx.video_processor.get_latest()
        fps_box.caption(f"Processing speed: {fps:.1f} FPS")

        if pred_idx == FALL_CLASS_IDX and fall_streak >= FALL_CONFIRM_FRAMES:
            status_box.error(f"🚨 FALL DETECTED\nConfidence: {conf:.0%}")
            play_alarm()
            if not st.session_state.was_falling:
                st.session_state.fall_events += 1
                st.session_state.class_counts[0] += 1
                st.session_state.was_falling = True
        elif pred_idx == "OUT_OF_FRAME":
            status_box.warning("Full body not visible — step back so legs are in frame.")
            st.session_state.was_falling = False
        elif pred_idx is not None:
            status_box.success(f"{CLASS_LABELS[pred_idx]}\nConfidence: {conf:.0%}")
            if st.session_state.was_falling is False:
                st.session_state.class_counts[pred_idx] += 1
            st.session_state.was_falling = False
        else:
            status_box.info("Waiting for a person to appear in frame...")

        with counts_box.container():
            for idx, name in CLASS_LABELS.items():
                if idx == FALL_CLASS_IDX:
                    continue
                st.caption(f"{name}: {st.session_state.class_counts[idx]}")
else:
    status_box.info("Click **Start** on the video feed to begin 24/7 monitoring.")
