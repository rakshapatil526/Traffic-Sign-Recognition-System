"""
Traffic Sign Recognition System - Procedural Sign & Road Scene Generator
Generates realistic traffic sign templates, realistic road images, and a dashcam drive video.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import math

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

def draw_traffic_sign(class_id: int, size: int = 128) -> np.ndarray:
    """
    Renders an authentic, crisp procedural traffic sign image for a given GTSRB class ID.
    Returns RGB image of shape (size, size, 3).
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 240  # light grey canvas
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    
    # 1. SPEED LIMIT SIGNS (0, 1, 2, 3, 4, 5, 7, 8)
    speed_map = {0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120"}
    if class_id in speed_map:
        # Red circular border (RGB: Red=230, Green=25, Blue=25)
        cv2.circle(img, center, radius, (230, 25, 25), -1)
        inner_radius = int(radius * 0.74)
        cv2.circle(img, center, inner_radius, (255, 255, 255), -1)  # White inner
        
        # Black numerals
        text = speed_map[class_id]
        scale = 1.05 if len(text) <= 2 else 0.8
        thickness = 3 if len(text) <= 2 else 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_size = cv2.getTextSize(text, font, scale, thickness)[0]
        t_x = center[0] - t_size[0] // 2
        t_y = center[1] + t_size[1] // 2
        cv2.putText(img, text, (t_x, t_y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        return img
        
    # 2. STOP SIGN (14) - Red Octagon
    if class_id == 14:
        r = radius
        angles = [i * 45 + 22.5 for i in range(8)]
        pts = []
        for a in angles:
            rad = math.radians(a)
            x = int(center[0] + r * math.cos(rad))
            y = int(center[1] + r * math.sin(rad))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], (225, 25, 25))
        
        # Inner white border
        r_in = int(radius * 0.92)
        pts_in = []
        for a in angles:
            rad = math.radians(a)
            pts_in.append([int(center[0] + r_in * math.cos(rad)), int(center[1] + r_in * math.sin(rad))])
        cv2.polylines(img, [np.array(pts_in, np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        
        # White "STOP" text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.85
        t_size = cv2.getTextSize("STOP", font, scale, 3)[0]
        cv2.putText(img, "STOP", (center[0] - t_size[0]//2, center[1] + t_size[1]//2), font, scale, (255, 255, 255), 3, cv2.LINE_AA)
        return img

    # 3. NO ENTRY (17) - Red circle with white horizontal bar
    if class_id == 17:
        cv2.circle(img, center, radius, (230, 25, 25), -1)
        bar_w = int(radius * 1.45)
        bar_h = int(radius * 0.36)
        x1 = center[0] - bar_w // 2
        y1 = center[1] - bar_h // 2
        cv2.rectangle(img, (x1, y1), (x1 + bar_w, y1 + bar_h), (255, 255, 255), -1)
        return img

    # 4. YIELD / GIVE WAY (13) - Inverted Triangle
    if class_id == 13:
        p1 = (center[0] - radius, center[1] - int(radius * 0.8))
        p2 = (center[0] + radius, center[1] - int(radius * 0.8))
        p3 = (center[0], center[1] + radius)
        tri_pts = np.array([p1, p2, p3], np.int32)
        cv2.fillPoly(img, [tri_pts], (230, 25, 25))
        
        # Inner white triangle
        p1_in = (center[0] - int(radius * 0.72), center[1] - int(radius * 0.65))
        p2_in = (center[0] + int(radius * 0.72), center[1] - int(radius * 0.65))
        p3_in = (center[0], center[1] + int(radius * 0.75))
        cv2.fillPoly(img, [np.array([p1_in, p2_in, p3_in], np.int32)], (255, 255, 255))
        return img

    # 5. TURN RIGHT / LEFT / MANDATORY (33: Turn Right, 34: Turn Left, 35: Ahead, 38: Keep Right, 40: Roundabout)
    if class_id in (33, 34, 35, 38, 40):
        # Deep blue circular background (RGB: R=25, G=90, B=225)
        cv2.circle(img, center, radius, (25, 90, 225), -1)
        cv2.circle(img, center, int(radius * 0.94), (255, 255, 255), 2, cv2.LINE_AA)
        
        if class_id == 33: # Turn Right
            p_start = (center[0] - int(radius * 0.35), center[1] + int(radius * 0.4))
            p_corner = (center[0] + int(radius * 0.15), center[1] + int(radius * 0.4))
            p_end = (center[0] + int(radius * 0.15), center[1] - int(radius * 0.2))
            cv2.arrowedLine(img, p_start, (center[0] + int(radius * 0.3), center[1] - int(radius * 0.1)), (255, 255, 255), 6, tipLength=0.4)
            cv2.arrowedLine(img, (center[0] - int(radius*0.3), center[1]), (center[0] + int(radius*0.35), center[1]), (255, 255, 255), 7, tipLength=0.45)
        elif class_id == 34: # Turn Left
            cv2.arrowedLine(img, (center[0] + int(radius*0.3), center[1]), (center[0] - int(radius*0.35), center[1]), (255, 255, 255), 7, tipLength=0.45)
        elif class_id == 35: # Straight / Ahead only
            cv2.arrowedLine(img, (center[0], center[1] + int(radius*0.4)), (center[0], center[1] - int(radius*0.4)), (255, 255, 255), 7, tipLength=0.4)
        elif class_id == 38: # Keep Right
            cv2.arrowedLine(img, (center[0] - int(radius*0.3), center[1] - int(radius*0.3)), (center[0] + int(radius*0.3), center[1] + int(radius*0.3)), (255, 255, 255), 7, tipLength=0.4)
        elif class_id == 40: # Roundabout
            for ang in [0, 120, 240]:
                rad_a = math.radians(ang)
                x = int(center[0] + radius * 0.45 * math.cos(rad_a))
                y = int(center[1] + radius * 0.45 * math.sin(rad_a))
                dx = int(-radius * 0.35 * math.sin(rad_a))
                dy = int(radius * 0.35 * math.cos(rad_a))
                cv2.arrowedLine(img, (x - dx//2, y - dy//2), (x + dx//2, y + dy//2), (255, 255, 255), 4, tipLength=0.5)
        return img

    # 6. WARNING / DANGER SIGNS (18: Danger, 19: Curve Left, 20: Curve Right, 25: Road Work, 27: Pedestrian)
    p1 = (center[0], center[1] - radius)
    p2 = (center[0] - int(radius * 1.05), center[1] + int(radius * 0.8))
    p3 = (center[0] + int(radius * 1.05), center[1] + int(radius * 0.8))
    cv2.fillPoly(img, [np.array([p1, p2, p3], np.int32)], (230, 25, 25))
    
    # Inner white triangle
    p1_in = (center[0], center[1] - int(radius * 0.72))
    p2_in = (center[0] - int(radius * 0.8), center[1] + int(radius * 0.65))
    p3_in = (center[0] + int(radius * 0.8), center[1] + int(radius * 0.65))
    cv2.fillPoly(img, [np.array([p1_in, p2_in, p3_in], np.int32)], (255, 255, 255))
    
    # Symbol inside warning triangle
    if class_id == 18:  # Exclamation mark
        cv2.line(img, (center[0], center[1] - int(radius * 0.3)), (center[0], center[1] + int(radius * 0.15)), (20, 20, 20), 6)
        cv2.circle(img, (center[0], center[1] + int(radius * 0.38)), 4, (20, 20, 20), -1)
    elif class_id in (19, 20):  # Curve Left / Right
        curve_dir = -1 if class_id == 19 else 1
        p_c1 = (center[0] - curve_dir * 10, center[1] + 15)
        p_c2 = (center[0] + curve_dir * 12, center[1] - 15)
        cv2.arrowedLine(img, p_c1, p_c2, (20, 20, 20), 5, tipLength=0.5)
    elif class_id == 25:  # Road work
        cv2.circle(img, (center[0], center[1] - 14), 5, (20, 20, 20), -1)
        cv2.line(img, (center[0], center[1] - 8), (center[0] + 6, center[1] + 12), (20, 20, 20), 4)
        cv2.line(img, (center[0] - 12, center[1] + 20), (center[0] + 16, center[1] + 20), (20, 20, 20), 4)
    else:  # General icon
        cv2.putText(img, "!", (center[0] - 8, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 4)
        
    return img

def create_realistic_road_scene(
    sign_specs: list,
    width: int = 1024,
    height: int = 680,
    scene_type: str = "highway"
) -> np.ndarray:
    """
    Renders a realistic road background with perspective asphalt, lane markings, horizon,
    sky, guardrails, roadside vegetation, and embeds given signs on support posts.
    
    sign_specs: List of dicts, e.g. [{"class_id": 3, "x": 780, "y": 280, "size": 130}]
    """
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    
    horizon_y = int(height * 0.44)
    
    # 1. Sky with realistic blue-to-horizon gradient
    for y in range(0, horizon_y):
        ratio = y / horizon_y
        b = int(140 + ratio * 60)
        g = int(180 + ratio * 40)
        r = int(220 - ratio * 40)
        scene[y, :] = (b, g, r)
        
    # Clouds
    cv2.ellipse(scene, (int(width * 0.25), int(horizon_y * 0.4)), (120, 30), 0, 0, 360, (245, 245, 250), -1)
    cv2.ellipse(scene, (int(width * 0.32), int(horizon_y * 0.35)), (80, 25), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(scene, (int(width * 0.72), int(horizon_y * 0.5)), (140, 35), 0, 0, 360, (240, 240, 245), -1)

    # 2. Distant mountains / trees along horizon
    for x in range(0, width, 8):
        m_h = int(25 * math.sin(x * 0.015) + 15 * math.cos(x * 0.035) + 30)
        scene[horizon_y - m_h:horizon_y, x:x+8] = (60, 110, 70)

    # 3. Roadside green terrain
    scene[horizon_y:, :] = (45, 120, 50)
    
    # 4. Asphalt road with perspective polygon
    vp_x = int(width * 0.5)
    road_pts = np.array([
        [vp_x - 35, horizon_y],
        [vp_x + 35, horizon_y],
        [width - 60, height],
        [60, height]
    ], np.int32)
    cv2.fillPoly(scene, [road_pts], (65, 68, 72))  # Dark asphalt
    
    # Road shoulders / gravel
    left_shoulder = np.array([[vp_x - 48, horizon_y], [vp_x - 35, horizon_y], [60, height], [15, height]], np.int32)
    right_shoulder = np.array([[vp_x + 35, horizon_y], [vp_x + 48, horizon_y], [width - 15, height], [width - 60, height]], np.int32)
    cv2.fillPoly(scene, [left_shoulder], (110, 115, 115))
    cv2.fillPoly(scene, [right_shoulder], (110, 115, 115))
    
    # White lane boundary lines
    cv2.line(scene, (vp_x - 35, horizon_y), (60, height), (230, 230, 230), 4, cv2.LINE_AA)
    cv2.line(scene, (vp_x + 35, horizon_y), (width - 60, height), (230, 230, 230), 4, cv2.LINE_AA)

    # Dashed yellow / white center divider lines
    num_dashes = 14
    for i in range(num_dashes):
        r1 = (i / num_dashes) ** 1.8
        r2 = ((i + 0.6) / num_dashes) ** 1.8
        y1 = int(horizon_y + r1 * (height - horizon_y))
        y2 = int(horizon_y + r2 * (height - horizon_y))
        x1 = int(vp_x + (y1 - horizon_y) * 0.02)
        x2 = int(vp_x + (y2 - horizon_y) * 0.02)
        thick = max(2, int(r2 * 7))
        cv2.line(scene, (x1, y1), (x2, y2), (240, 240, 240), thick, cv2.LINE_AA)

    # 5. Embed Traffic Signs on Posts
    for spec in sign_specs:
        cid = spec["class_id"]
        sx = spec["x"]
        sy = spec["y"]
        ssize = spec["size"]
        
        # Draw metal mounting post / pole extending to ground
        pole_bottom_y = min(height - 40, sy + ssize + 160)
        cv2.line(scene, (sx + ssize//2, sy + ssize//2), (sx + ssize//2, pole_bottom_y), (140, 145, 150), 7)
        cv2.line(scene, (sx + ssize//2 - 1, sy + ssize//2), (sx + ssize//2 - 1, pole_bottom_y), (190, 195, 200), 2)
        
        # Render sign graphic
        sign_img = draw_traffic_sign(cid, size=ssize)
        
        # Add slight realistic vignetting / shadow
        y1 = max(0, sy)
        y2 = min(height, sy + ssize)
        x1 = max(0, sx)
        x2 = min(width, sx + ssize)
        
        # Render sign graphic
        sign_img = draw_traffic_sign(cid, size=ssize)
        
        y1 = max(0, sy)
        y2 = min(height, sy + ssize)
        x1 = max(0, sx)
        x2 = min(width, sx + ssize)
        
        sub_sign = sign_img[0:(y2-y1), 0:(x2-x1)]
        # Background canvas is initialized to (240, 240, 240)
        # Any pixel not equal to (240, 240, 240) belongs to the traffic sign!
        mask = (sub_sign[:, :, 0] != 240) | (sub_sign[:, :, 1] != 240) | (sub_sign[:, :, 2] != 240)
        
        # Realistic circular / shaped drop shadow behind sign
        shadow_off = 4
        shadow_y1 = min(height, y1 + shadow_off)
        shadow_y2 = min(height, y2 + shadow_off)
        shadow_x1 = min(width, x1 + shadow_off)
        shadow_x2 = min(width, x2 + shadow_off)
        
        sh_h = shadow_y2 - shadow_y1
        sh_w = shadow_x2 - shadow_x1
        if sh_h > 0 and sh_w > 0:
            sh_mask = mask[0:sh_h, 0:sh_w]
            sh_roi = scene[shadow_y1:shadow_y2, shadow_x1:shadow_x2]
            sh_roi[sh_mask] = (30, 32, 36)
            scene[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = sh_roi

        # Blend sign onto road scene
        roi = scene[y1:y2, x1:x2]
        roi[mask] = sub_sign[mask]
        scene[y1:y2, x1:x2] = roi

    return scene

def generate_sample_images():
    """Generates a rich suite of realistic sample road traffic images."""
    samples = [
        {
            "filename": "sample_highway_speed60.jpg",
            "title": "Highway Arterial - Speed Limit 60 km/h",
            "specs": [{"class_id": 3, "x": 760, "y": 230, "size": 130}]
        },
        {
            "filename": "sample_urban_stop_sign.jpg",
            "title": "Urban Intersection - Stop Sign",
            "specs": [{"class_id": 14, "x": 770, "y": 240, "size": 135}]
        },
        {
            "filename": "sample_one_way_no_entry.jpg",
            "title": "Downtown Lane - No Entry Restriction",
            "specs": [{"class_id": 17, "x": 750, "y": 220, "size": 140}]
        },
        {
            "filename": "sample_roundabout_ahead.jpg",
            "title": "Roundabout Approach - Mandatory Roundabout",
            "specs": [
                {"class_id": 40, "x": 770, "y": 230, "size": 130},
                {"class_id": 13, "x": 160, "y": 260, "size": 105}
            ]
        },
        {
            "filename": "sample_mountain_warning.jpg",
            "title": "Mountain Highway - Curve & Caution Warning",
            "specs": [
                {"class_id": 18, "x": 760, "y": 230, "size": 135},
                {"class_id": 33, "x": 160, "y": 270, "size": 105}
            ]
        },
        {
            "filename": "sample_school_zone_speed30.jpg",
            "title": "School Zone - Speed Limit 30 km/h & Work Ahead",
            "specs": [
                {"class_id": 1, "x": 760, "y": 220, "size": 135},
                {"class_id": 25, "x": 160, "y": 260, "size": 110}
            ]
        }
    ]
    
    generated_paths = []
    for s in samples:
        path = os.path.join(SAMPLES_DIR, s["filename"])
        img = create_realistic_road_scene(s["specs"])
        # Convert RGB to BGR for OpenCV imwrite
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        generated_paths.append(path)
        print(f"Generated sample image: {path}")
        
    return generated_paths

def generate_sample_traffic_video():
    """
    Renders an animated 120-frame (8-second) driving dashcam video passing
    multiple traffic signs (Speed Limit 60, Stop Sign, Turn Right) smoothly.
    Saved to data/samples/sample_traffic_drive.mp4 (and .avi fallback).
    """
    video_path_mp4 = os.path.join(SAMPLES_DIR, "sample_traffic_drive.mp4")
    video_path_avi = os.path.join(SAMPLES_DIR, "sample_traffic_drive.avi")
    
    width = 720
    height = 480
    fps = 15
    total_frames = 120
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_mp4, fourcc, fps, (width, height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path_avi, fourcc, fps, (width, height))
        video_target = video_path_avi
    else:
        video_target = video_path_mp4

    print("Rendering sample drive video sequence...")
    for frame_idx in range(total_frames):
        # Progress from 0 to 1
        t = frame_idx / total_frames
        
        # Sign 1: Speed Limit 60 (frames 0 to 45)
        # Sign 2: Stop Sign (frames 40 to 85)
        # Sign 3: Turn Right Ahead (frames 80 to 120)
        specs = []
        
        if frame_idx < 45:
            p = frame_idx / 45.0
            size = int(45 + p * 110)
            x = int(width * 0.58 + p * (width * 0.28))
            y = int(height * 0.45 - p * 50)
            specs.append({"class_id": 3, "x": x, "y": y, "size": size})
        elif frame_idx < 85:
            p = (frame_idx - 40) / 45.0
            size = int(45 + p * 115)
            x = int(width * 0.58 + p * (width * 0.28))
            y = int(height * 0.45 - p * 50)
            specs.append({"class_id": 14, "x": x, "y": y, "size": size})
        else:
            p = (frame_idx - 80) / 40.0
            size = int(45 + p * 110)
            x = int(width * 0.58 + p * (width * 0.28))
            y = int(height * 0.45 - p * 50)
            specs.append({"class_id": 33, "x": x, "y": y, "size": size})
            
        frame_rgb = create_realistic_road_scene(specs, width=width, height=height)
        
        # Add realistic dashcam HUD overlay
        timestamp_str = f"2026-11-01 14:{30 + frame_idx//60:02d}:{frame_idx%60:02d} | 62 km/h"
        cv2.putText(frame_rgb, timestamp_str, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_rgb, "ADAS CAM 01 [FRONT]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)
        
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    print(f"Generated sample drive video: {video_target}")
    return video_target

if __name__ == "__main__":
    generate_sample_images()
    generate_sample_traffic_video()
