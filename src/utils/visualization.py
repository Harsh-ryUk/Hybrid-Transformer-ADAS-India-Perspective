import cv2
import numpy as np

COLOR_LANE_LEFT = (0, 0, 255)       # Red
COLOR_LANE_RIGHT = (255, 0, 0)      # Blue
COLOR_LANE_CENTER = (0, 255, 0)     # Green
COLOR_DAMAGE = (0, 165, 255)        # Orange (BGR)
COLOR_TEXT = (255, 255, 255)        # White
COLOR_BG_DARK = (40, 40, 40)        # Dark Gray for dashboard

def draw_dashboard(frame, fps, latency, system_status="OK"):
    """
    Draws a dashboard overlay on top of the frame.
    """
    height, width = frame.shape[:2]
    
    # Create top bar
    bar_height = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), COLOR_BG_DARK, -1)
    
    # Apply transparency
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Text info
    info_text = f"FPS: {fps:.1f} | Latency: {latency:.1f}ms | System: {system_status} | ADAS Active"
    
    cv2.putText(frame, info_text, (20, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    return frame

def draw_lanes(frame, left_line, right_line, is_deep=False):
    """
    Draws lane lines on the frame.
    left_line, right_line: List of points or coefficients depending on implementation.
    Here we assume they are simple point lists [(x1, y1), (x2, y2)] representing the line segment
    OR polynomial points.
    """
    overlay = frame.copy()
    
    if left_line is not None:
        cv2.line(overlay, left_line[0], left_line[1], COLOR_LANE_LEFT, 5)
        
    if right_line is not None:
        cv2.line(overlay, right_line[0], right_line[1], COLOR_LANE_RIGHT, 5)
        
    # Draw transparent polygon if both exist
    if left_line is not None and right_line is not None:
        pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    
    # Apply weighted add for transparency
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    label = "Deep Lane" if is_deep else "Classic Lane"
    cv2.putText(frame, label, (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return frame

def draw_damage(frame, boxes, scores, classes, class_names):
    """
    Draws bounding boxes for road damage.
    boxes: list of [x1, y1, x2, y2]
    """
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DAMAGE, 2)
        
        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), COLOR_DAMAGE, -1)
        
        # Text
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return frame

def draw_poly_lane(frame, left_poly_pts, right_poly_pts):
    """
    Draws polynomial fitted lanes (for deep learning output or advanced classical).
    pts: Array of shapes (N, 2)
    """
    if left_poly_pts is None and right_poly_pts is None:
        return frame
        
    overlay = frame.copy()
    
    if left_poly_pts is not None:
        cv2.polylines(overlay, [left_poly_pts], False, COLOR_LANE_LEFT, 5)
    
    if right_poly_pts is not None:
        cv2.polylines(overlay, [right_poly_pts], False, COLOR_LANE_RIGHT, 5)
        
    if left_poly_pts is not None and right_poly_pts is not None:
        # Create a single polygon for the lane area
         # We need to reverse right_poly_pts to form a closed loop properly
        pts = np.vstack((left_poly_pts, right_poly_pts[::-1]))
        cv2.fillPoly(overlay, [pts], COLOR_LANE_CENTER)
        
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame
