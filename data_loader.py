import cv2
import numpy as np
import os
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import mediapipe as mp
import pandas as pd
from pathlib import Path

class VideoFeatureExtractor:
    def __init__(self, pose_model_path='pose_landmarker_lite.task'):
        """Initialize the pose landmarker for feature extraction."""
        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"Pose model not found: {pose_model_path}")
        
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=pose_model_path),
                min_pose_detection_confidence=0.5
            )
        )
    
    @staticmethod
    def calc_angle(a, b, c):
        """Calculate angle between three points."""
        a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
        rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        ang = np.abs(rad * 180.0 / np.pi)
        return 360 - ang if ang > 180.0 else ang
    
    @staticmethod
    def extract_features_from_landmarks(landmarks):
        """Extract invariant features from pose landmarks."""
        if not landmarks:
            return None
        
        landmarks = landmarks[0]  # Get first person
        
        nose = landmarks[0]; l_sh = landmarks[11]; r_sh = landmarks[12]
        l_el = landmarks[13]; r_el = landmarks[14]
        l_wr = landmarks[15]; r_wr = landmarks[16]
        l_hip = landmarks[23]; r_hip = landmarks[24]
        l_knee = landmarks[25]; r_knee = landmarks[26]
        l_ankle = landmarks[27]; r_ankle = landmarks[28]
        l_foot = landmarks[31]; r_foot = landmarks[32]
        
        head_tilt = VideoFeatureExtractor.calc_angle(l_sh, nose, r_sh)
        l_sh_angle = VideoFeatureExtractor.calc_angle(r_sh, l_sh, nose)
        r_sh_angle = VideoFeatureExtractor.calc_angle(l_sh, r_sh, nose)
        l_elbow = VideoFeatureExtractor.calc_angle(l_sh, l_el, l_wr)
        r_elbow = VideoFeatureExtractor.calc_angle(r_sh, r_el, r_wr)
        
        if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
            l_hip_angle = VideoFeatureExtractor.calc_angle(l_sh, l_hip, l_knee)
            r_hip_angle = VideoFeatureExtractor.calc_angle(r_sh, r_hip, r_knee)
            l_knee_angle = VideoFeatureExtractor.calc_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = VideoFeatureExtractor.calc_angle(r_hip, r_knee, r_ankle)
            l_ankle_angle = VideoFeatureExtractor.calc_angle(l_knee, l_ankle, l_foot)
            r_ankle_angle = VideoFeatureExtractor.calc_angle(r_knee, r_ankle, r_foot)
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
                        sh_line_angle, hip_line_angle, shoulder_width, leg_torso_ratio])
    
    def extract_from_video(self, video_path, annotation_file):
        """Extract features from a video and return labeled data."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        # Parse annotation file to get fall frames
        fall_frames = set()
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    parts = lines[0].strip().split()
                    if len(parts) >= 2:
                        start_frame = int(parts[0])
                        end_frame = int(parts[1])
                        fall_frames = set(range(start_frame, end_frame + 1))
        except:
            pass
        
        data = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
            
            if result.pose_landmarks:
                features = self.extract_features_from_landmarks(result.pose_landmarks)
                if features is not None:
                    # Label: 0 = Fall, 1 = Normal
                    label = 0 if frame_count in fall_frames else 1
                    data.append({
                        'features': features,
                        'label': label,
                        'frame': frame_count,
                        'video': video_path
                    })
            
            frame_count += 1
        
        cap.release()
        return data


def load_dataset(data_root, pose_model_path='pose_landmarker_lite.task', max_videos_per_location=5):
    """
    Load dataset from the LE2I dataset structure.
    
    Args:
        data_root: Path to extracted dataset root
        pose_model_path: Path to MediaPipe pose model
        max_videos_per_location: Maximum videos to load per location (for efficiency)
    """
    extractor = VideoFeatureExtractor(pose_model_path)
    all_data = []
    
    locations = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for location in sorted(locations):
        location_path = os.path.join(data_root, location, location)
        video_dir = os.path.join(location_path, 'Videos')
        annotation_dir = os.path.join(location_path, 'Annotation_files')
        
        if not os.path.exists(video_dir):
            print(f"Skipping {location}: Videos folder not found")
            continue
        
        videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.avi')])
        
        for i, video_file in enumerate(videos[:max_videos_per_location]):
            video_path = os.path.join(video_dir, video_file)
            annotation_file = os.path.join(annotation_dir, video_file.replace('.avi', '.txt'))
            
            print(f"Processing {location} - {video_file} ({i+1}/{min(len(videos), max_videos_per_location)})")
            
            try:
                data = extractor.extract_from_video(video_path, annotation_file)
                if data:
                    all_data.extend(data)
                    print(f"  ✓ Extracted {len(data)} frames")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    return all_data
