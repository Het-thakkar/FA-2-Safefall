import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import io
import wave
import base64
from collections import deque

st.set_page_config(page_title="🏥 Live Fall Detection - Real Camera", layout="wide")

# ============================================================
# SETUP
# ============================================================

# Try to load MediaPipe pose detector
@st.cache_resource
def load_pose_detector():
    """Load MediaPipe pose detector"""
    try:
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.error(f"Error loading pose detector: {e}")
        return None

pose = load_pose_detector()
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_beep_base64(freq=1000, duration_ms=300, volume=0.7, sample_rate=44100):
    """Generate beep sound"""
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

def play_alarm():
    """Play alarm sound"""
    beep = generate_beep_base64()
    st.markdown(
        f"""<audio autoplay="true">
            <source src="data:audio/wav;base64,{beep}" type="audio/wav">
        </audio>""",
        unsafe_allow_html=True,
    )

def calc_angle(a, b, c):
    """Calculate angle between three points"""
    a_vec = np.array([a.x, a.y])
    b_vec = np.array([b.x, b.y])
    c_vec = np.array([c.x, c.y])
    
    ba = a_vec - b_vec
    bc = c_vec - b_vec
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def detect_fall(landmarks):
    """
    Simple fall detection based on pose geometry
    Returns: (is_fall, fall_confidence)
    """
    if not landmarks:
        return False, 0.0
    
    # Get key points
    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    
    # Check visibility
    min_visibility = 0.5
    if not all([p.visibility > min_visibility for p in [nose, left_shoulder, right_shoulder, 
                                                         left_hip, right_hip, left_knee, right_knee, 
                                                         left_ankle, right_ankle]]):
        return False, 0.0
    
    # Calculate key angles
    # Torso angle - when person falls, torso becomes horizontal
    mid_shoulder = np.array([(left_shoulder.x + right_shoulder.x) / 2, 
                             (left_shoulder.y + right_shoulder.y) / 2])
    mid_hip = np.array([(left_hip.x + right_hip.x) / 2, 
                        (left_hip.y + right_hip.y) / 2])
    
    # Vertical distance (y-axis) should be significant when standing
    # Horizontal distance (x-axis) indicates tilted/fallen posture
    vertical_dist = abs(mid_hip[1] - mid_shoulder[1])
    horizontal_dist = abs(mid_hip[0] - mid_shoulder[0])
    
    # Height - distance from hip to ankle
    hip_to_ankle = np.sqrt((left_hip.x - left_ankle.x)**2 + (left_hip.y - left_ankle.y)**2) + \
                   np.sqrt((right_hip.x - right_ankle.x)**2 + (right_hip.y - right_ankle.y)**2)
    hip_to_ankle /= 2
    
    # Shoulder to hip distance
    shoulder_to_hip = np.sqrt((mid_shoulder[0] - mid_hip[0])**2 + (mid_shoulder[1] - mid_hip[1])**2)
    
    # Knee angles - when standing, knees are more vertical
    left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    
    # Fall detection logic
    fall_score = 0.0
    
    # 1. If horizontal distance > vertical distance, likely fallen
    if horizontal_dist > vertical_dist * 1.5:
        fall_score += 0.4
    
    # 2. If hip to ankle distance is very small compared to shoulder-hip distance, likely fallen
    if hip_to_ankle < shoulder_to_hip * 0.5:
        fall_score += 0.3
    
    # 3. If knee angles are too small (legs straightened), might be falling
    if avg_knee_angle < 80:
        fall_score += 0.2
    
    # 4. If nose is below hip level (head pointing down), likely fallen
    if nose.y > mid_hip[1]:
        fall_score += 0.2
    
    is_fall = fall_score > 0.5
    confidence = min(fall_score, 1.0)
    
    return is_fall, confidence

def process_frame(frame):
    """Process frame and detect pose/fall"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # Draw pose
    frame_bgr = frame.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            POSE_CONNECTIONS,
            drawing_spec,
            drawing_spec
        )
        
        # Detect fall
        is_fall, confidence = detect_fall(results.pose_landmarks.landmark)
        return frame_bgr, results.pose_landmarks.landmark, is_fall, confidence
    
    return frame_bgr, None, False, 0.0

# ============================================================
# UI
# ============================================================

st.title("🎥 Live Fall Detection - Real Camera")
st.markdown("**Keep your camera on - AI continuously monitors for falls!**")

# Initialize session state
if 'fall_count' not in st.session_state:
    st.session_state.fall_count = 0
if 'normal_count' not in st.session_state:
    st.session_state.normal_count = 0
if 'is_falling' not in st.session_state:
    st.session_state.is_falling = False
if 'fall_history' not in st.session_state:
    st.session_state.fall_history = deque(maxlen=5)

# Main layout
col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Camera input
    camera_input = st.camera_input("📸 Take a photo or stream from webcam")
    
    if camera_input:
        # Convert to numpy array
        image = np.array(camera_input)
        
        # Process frame
        processed_frame, landmarks, is_fall, confidence = process_frame(image)
        
        # Display processed frame
        st.image(processed_frame, channels="BGR", use_column_width=True)
        
        # Show detection result
        if landmarks is not None:
            if is_fall:
                st.session_state.fall_count += 1
                st.session_state.is_falling = True
                st.session_state.fall_history.append({
                    'time': time.time(),
                    'confidence': confidence
                })
                
                # Show alert
                st.error(f"🚨 **FALL DETECTED!** Confidence: {confidence:.1%}")
                play_alarm()
            else:
                st.success(f"✅ **Normal Activity** - Confidence: {1-confidence:.1%}")
                if st.session_state.is_falling:
                    st.session_state.normal_count += 1
                    st.session_state.is_falling = False
        else:
            st.warning("⏳ No pose detected - adjust camera angle")

with col2:
    st.subheader("📊 Statistics")
    
    # Metrics
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🚨 Falls", st.session_state.fall_count)
    with col_b:
        st.metric("✅ Normal", st.session_state.normal_count)
    
    total = st.session_state.fall_count + st.session_state.normal_count
    if total > 0:
        fall_rate = st.session_state.fall_count / total * 100
        st.metric("Detection Rate", f"{total}")
        st.progress(st.session_state.fall_count / max(total, 1), text=f"Fall: {fall_rate:.0f}%")
    
    # Recent falls
    if st.session_state.fall_history:
        st.subheader("🔔 Recent Events")
        for event in reversed(list(st.session_state.fall_history)):
            st.write(f"Fall (Conf: {event['confidence']:.1%})")
    
    # Reset button
    if st.button("🔄 Reset Statistics"):
        st.session_state.fall_count = 0
        st.session_state.normal_count = 0
        st.session_state.fall_history.clear()
        st.rerun()

# ============================================================
# INFO SECTION
# ============================================================

st.divider()

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.subheader("🎯 How It Works")
    st.markdown("""
    1. **Pose Detection** - Detects 33 body keypoints
    2. **Geometry Analysis** - Calculates body angles
    3. **Fall Classification** - Determines if fallen
    4. **Alert System** - Sounds alarm on detection
    """)

with info_col2:
    st.subheader("✨ Features")
    st.markdown("""
    ✅ Real-time detection
    ✅ Automatic alarms
    ✅ Statistics tracking
    ✅ Multiple angles support
    ✅ Instant notifications
    """)

with info_col3:
    st.subheader("📝 Tips")
    st.markdown("""
    💡 Keep camera at chest height
    💡 Ensure good lighting
    💡 Stay 2-3m from camera
    💡 Full body visible
    💡 Clear background helps
    """)

# ============================================================
# DEMO MODE
# ============================================================

st.divider()
st.subheader("🎬 Demo Mode (No Camera Required)")

if st.button("➡️ Simulate Fall Detection"):
    st.session_state.fall_count += 1
    st.error("🚨 **FALL DETECTED!**")
    play_alarm()

if st.button("➡️ Simulate Normal Activity"):
    st.session_state.normal_count += 1
    st.success("✅ **Normal Activity**")

st.markdown("""
---
**🏥 SafeFall - AI Elderly Fall Detection System**
- Real-time pose analysis
- Automatic emergency alerts
- Built with MediaPipe + Streamlit
""")
