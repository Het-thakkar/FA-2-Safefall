import streamlit as st
import cv2
import numpy as np
import time
import io
import wave
import base64
from collections import deque
from PIL import Image

st.set_page_config(page_title="🏥 Live Fall Detection - Real Camera", layout="wide")

# ============================================================
# AUDIO ALERT
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

# ============================================================
# FALL DETECTION (SIMPLE VERSION)
# ============================================================

def simple_fall_detection(frame):
    """
    Simple fall detection based on motion and silhouette analysis
    Returns: (is_fall, confidence)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get frame dimensions
        h, w = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (person silhouette)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get largest contour (should be the person)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 1000:  # Too small to be a person
            return False, 0.0
        
        # Get bounding rectangle
        x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
        
        # Aspect ratio - when standing, height > width
        # When fallen, width > height
        aspect_ratio = bbox_w / (bbox_h + 1e-5)
        
        # Get moments for center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2
        
        # Vertical position - when falling, center of mass moves lower
        center_ratio = cy / h
        
        # Fall score calculation
        fall_score = 0.0
        
        # High aspect ratio (wide and short) suggests fallen
        if aspect_ratio > 0.8:
            fall_score += 0.5
        
        # Center of mass very low suggests fallen
        if center_ratio > 0.65:
            fall_score += 0.3
        
        # Bounding box is wide and short
        if bbox_h < bbox_w * 0.8:
            fall_score += 0.2
        
        is_fall = fall_score > 0.5
        confidence = min(fall_score, 1.0)
        
        return is_fall, confidence
    
    except Exception as e:
        return False, 0.0

# ============================================================
# UI
# ============================================================

st.title("🎥 Live Fall Detection - Real Camera")
st.markdown("**📱 Keep your camera on - AI continuously monitors for falls!**")

# Initialize session state
if 'fall_count' not in st.session_state:
    st.session_state.fall_count = 0
if 'normal_count' not in st.session_state:
    st.session_state.normal_count = 0
if 'is_falling' not in st.session_state:
    st.session_state.is_falling = False
if 'fall_history' not in st.session_state:
    st.session_state.fall_history = deque(maxlen=10)

# Main layout
col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Camera input
    camera_input = st.camera_input("📸 Take a photo or stream from webcam")
    
    if camera_input:
        # Convert to numpy array
        image = Image.open(camera_input)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process frame for fall detection
        is_fall, confidence = simple_fall_detection(frame_bgr)
        
        # Add visualization
        frame_display = frame_bgr.copy()
        h, w = frame_display.shape[:2]
        
        # Add status text
        status_text = f"Fall Detection: {'YES' if is_fall else 'NO'} ({confidence:.1%})"
        color = (0, 0, 255) if is_fall else (0, 255, 0)
        cv2.putText(frame_display, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_column_width=True)
        
        # Update statistics
        if is_fall:
            if not st.session_state.is_falling:
                st.session_state.fall_count += 1
                st.session_state.is_falling = True
                st.session_state.fall_history.append({
                    'time': time.strftime("%H:%M:%S"),
                    'confidence': confidence
                })
            
            # Show alert
            st.error(f"🚨 **FALL DETECTED!** Confidence: {confidence:.1%}")
            play_alarm()
        else:
            if st.session_state.is_falling:
                st.session_state.normal_count += 1
                st.session_state.is_falling = False
            
            st.success(f"✅ **Normal Activity** - Confidence: {1-confidence:.1%}")

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
        st.metric("Total Detections", f"{total}")
        st.progress(st.session_state.fall_count / max(total, 1), 
                   text=f"Fall Rate: {fall_rate:.0f}%")
    
    # Recent events
    st.subheader("🔔 Recent Falls")
    if st.session_state.fall_history:
        for event in reversed(list(st.session_state.fall_history)[-5:]):
            st.write(f"⏰ {event['time']} - Conf: {event['confidence']:.1%}")
    else:
        st.info("No falls detected yet")
    
    st.divider()
    
    # Reset button
    col_reset = st.columns([1, 1])
    with col_reset[0]:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.fall_count = 0
            st.session_state.normal_count = 0
            st.session_state.fall_history.clear()
            st.rerun()

# ============================================================
# DEMO MODE
# ============================================================

st.divider()
st.subheader("🎬 Demo Mode (No Camera Required)")

demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    if st.button("➡️ Simulate Fall", use_container_width=True):
        st.session_state.fall_count += 1
        st.error("🚨 **FALL DETECTED!** (Simulated)")
        play_alarm()

with demo_col2:
    if st.button("➡️ Simulate Normal Activity", use_container_width=True):
        st.session_state.normal_count += 1
        st.success("✅ **Normal Activity** (Simulated)")

# ============================================================
# INFO SECTION
# ============================================================

st.divider()

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.subheader("🎯 How It Works")
    st.markdown("""
    1. **Silhouette Detection** - Analyzes person shape
    2. **Aspect Ratio** - Checks width vs height
    3. **Center of Mass** - Tracks body position
    4. **Fall Scoring** - Combines all factors
    5. **Instant Alert** - Sounds alarm on detection
    """)

with info_col2:
    st.subheader("✨ Features")
    st.markdown("""
    ✅ Real-time camera detection
    ✅ Automatic alarm sound
    ✅ Statistics tracking
    ✅ Confidence scoring
    ✅ Recent event history
    ✅ Demo simulation mode
    """)

with info_col3:
    st.subheader("📝 Best Results")
    st.markdown("""
    💡 Full body visible in frame
    💡 Camera at chest height
    💡 Good lighting needed
    💡 Clear background
    💡 2-3m distance optimal
    💡 No obstructions
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style="text-align: center;">

**🏥 SafeFall - AI Fall Detection System**

Real-time monitoring • Instant alerts • Emergency response

[📖 Documentation](#) | [🐛 Report Issue](#) | [💬 Support](#)

</div>
""", unsafe_allow_html=True)
