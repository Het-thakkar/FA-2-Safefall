import streamlit as st
import cv2, os, io, wave, base64, threading, time
import numpy as np
import mediapipe as mp
import joblib
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="SafeFall AI - CCTV Live Monitor", layout="wide")

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
            min_pose_detection_confidence=0.5
        )
    )

rf_model = load_rf_model()
scaler = load_scaler()
landmarker = load_landmarker()

CLASS_LABELS = {0: "FALL DETECTED", 1: "Walking", 2: "Sitting", 3: "Standing", 4: "Normal Activity"}
CLASS_COLORS_BGR = {0: (0, 0, 255), 1: (255, 165, 0), 2: (255, 0, 255), 3: (0, 255, 0), 4: (200, 200, 200)}
FALL_CLASS_IDX = 0
FALL_CONFIRM_FRAMES = 5  # consecutive fall-frames required before alarm fires (avoids single-frame false alarms)
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
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    annotated = draw_landmarks_on_image(img_rgb.copy(), result)
    if not result.pose_landmarks:
        return annotated, None, 0.0

    landmarks = result.pose_landmarks[0]

    # --- Visibility guard ---
    # If hips/knees/ankles aren't actually visible (e.g. camera only shows head
    # and shoulders), the model has no real leg information to work with and
    # will guess -- often wrongly. Refuse to classify instead of guessing.
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_knee, r_knee = landmarks[25], landmarks[26]
    l_ankle, r_ankle = landmarks[27], landmarks[28]
    lower_body_visible = all(
        lm.visibility > 0.5 for lm in [l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
    )
    if not lower_body_visible:
        return annotated, "OUT_OF_FRAME", 0.0

    feats_raw = extract_invariant_features(landmarks)
    feats_scaled = scaler.transform(feats_raw)
    probs = rf_model.predict_proba(feats_scaled)
    pred_idx = int(np.argmax(probs[0]))
    conf = float(np.max(probs[0]))

    # --- Safety override ---
    # If the model says "Fall" but the leg geometry clearly indicates bent knees
    # (sitting posture) and the hips aren't near the bottom of the frame (i.e.
    # not actually on the ground), this is very likely a misclassified sit, not
    # a real fall. Downgrade it rather than trigger a false alarm.
    if pred_idx == FALL_CLASS_IDX:
        l_knee_angle = calc_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calc_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2  # 0 = top of frame, 1 = bottom

        looks_like_seated_not_fallen = avg_knee_angle < 130 and mid_hip_y < 0.75
        if looks_like_seated_not_fallen:
            pred_idx = 2  # Sitting
            conf = min(conf, 0.6)  # reflect the uncertainty of this override

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
# Continuous video processor -- this is the "CCTV" part: it keeps
# running frame after frame for as long as the stream is active,
# with no need to click anything again after pressing Start once.
# ---------------------------------------------------------------
class ContinuousMonitor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_pred_idx = None
        self.latest_conf = 0.0
        self.last_infer_time = 0
        self.fall_streak = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        now = time.time()
        if (now - self.last_infer_time) > 0.12:  # ~8 inferences/sec, keeps video smooth
            self.last_infer_time = now
            annotated_rgb, pred_idx, conf = run_prediction(img_rgb)

            with self.lock:
                self.latest_pred_idx = pred_idx
                self.latest_conf = conf
                if pred_idx == FALL_CLASS_IDX:
                    self.fall_streak += 1
                else:
                    self.fall_streak = 0

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Draw on-screen label directly on the video feed (visible even without the sidebar)
            if pred_idx == "OUT_OF_FRAME":
                cv2.putText(annotated_bgr, "Move back - full body not in frame", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            elif pred_idx is not None:
                label = CLASS_LABELS[pred_idx]
                color = CLASS_COLORS_BGR[pred_idx]
                if pred_idx == FALL_CLASS_IDX and self.fall_streak >= FALL_CONFIRM_FRAMES:
                    cv2.rectangle(annotated_bgr, (0, 0), (annotated_bgr.shape[1], annotated_bgr.shape[0]), color, 10)
                    cv2.putText(annotated_bgr, f"FALL DETECTED! ({conf:.0%})", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                else:
                    cv2.putText(annotated_bgr, f"{label} ({conf:.0%})", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            annotated_bgr = img

        return frame.from_ndarray(annotated_bgr, format="bgr24")

    def get_latest(self):
        with self.lock:
            return self.latest_pred_idx, self.latest_conf, self.fall_streak


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🎥 SafeFall AI — Continuous Live Monitor")
st.caption(
    "Turn the camera on once — it keeps detecting continuously (Fall / Walking / Sitting / "
    "Standing / Normal) with no need to click anything again, like a CCTV feed."
)

if 'fall_events' not in st.session_state:
    st.session_state.fall_events = 0
if 'was_falling' not in st.session_state:
    st.session_state.was_falling = False

col_video, col_status = st.columns([3, 1])

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with col_video:
    ctx = webrtc_streamer(
        key="safefall-cctv",
        video_processor_factory=ContinuousMonitor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

with col_status:
    status_box = st.empty()
    st.metric("Fall Events", st.session_state.fall_events)

if ctx.state.playing:
    st_autorefresh(interval=800, key="cctv_refresh")  # polls processor ~1.25x/sec to update sidebar + alarm

    if ctx.video_processor:
        pred_idx, conf, fall_streak = ctx.video_processor.get_latest()

        if pred_idx == FALL_CLASS_IDX and fall_streak >= FALL_CONFIRM_FRAMES:
            status_box.error(f"🚨 FALL DETECTED\nConfidence: {conf:.0%}")
            play_alarm()
            if not st.session_state.was_falling:
                st.session_state.fall_events += 1
                st.session_state.was_falling = True
        elif pred_idx == "OUT_OF_FRAME":
            status_box.warning("Full body not visible — step back so your legs are in frame.")
            st.session_state.was_falling = False
        elif pred_idx is not None:
            status_box.success(f"{CLASS_LABELS[pred_idx]}\nConfidence: {conf:.0%}")
            st.session_state.was_falling = False
        else:
            status_box.info("Waiting for a person to appear in frame...")
else:
    status_box.info("Click **Start** on the video feed to begin continuous monitoring.")
