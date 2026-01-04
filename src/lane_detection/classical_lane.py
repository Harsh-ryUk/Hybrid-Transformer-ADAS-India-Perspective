import cv2
import numpy as np

class ClassicalLaneDetector:
    def __init__(self):
        self.prev_left_fit = None
        self.prev_right_fit = None

    def process(self, frame):
        """
        Process frame to find lane lines.
        Returns:
            left_line: ((x1, y1), (x2, y2))
            right_line: ((x1, y1), (x2, y2))
        """
        height, width = frame.shape[:2]
        
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Region of Interest Mask
        # Assumption: Camera is mounted on dashboard center
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width // 2 - 50, height // 2 + 50),
            (width // 2 + 50, height // 2 + 50),
            (width, height)
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 5. Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )
        
        # 6. Average Lines
        left_line, right_line = self.average_slope_intercept(frame, lines)
        
        return left_line, right_line

    def average_slope_intercept(self, frame, lines):
        """
        Aggregates Hough lines into a single left and right line.
        """
        left_fit = []
        right_fit = []
        if lines is None:
            return None, None
            
        height, width = frame.shape[:2]
        center_x = width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            # Filter by slope to ignore horizontalish lines
            # Left lane: negative slope
            # Right lane: positive slope
            if slope < -0.5 and x1 < center_x and x2 < center_x:
                left_fit.append((slope, intercept))
            elif slope > 0.5 and x1 > center_x and x2 > center_x:
                right_fit.append((slope, intercept))
        
        left_line = self.make_coordinates(frame, np.average(left_fit, axis=0)) if left_fit else None
        right_line = self.make_coordinates(frame, np.average(right_fit, axis=0)) if right_fit else None
        
        return left_line, right_line

    def make_coordinates(self, frame, line_parameters):
        if line_parameters is None or np.isnan(line_parameters).any():
            return None
            
        slope, intercept = line_parameters
        height = frame.shape[0]
        
        y1 = height
        y2 = int(height * 0.6) # Draw up to 60% of screen height
        
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return ((x1, y1), (x2, y2))
        except OverflowError:
            return None
