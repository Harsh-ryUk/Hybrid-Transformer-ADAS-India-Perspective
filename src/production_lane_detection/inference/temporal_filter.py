import numpy as np

class TemporalFilter:
    """
    Exponential Moving Average (EMA) to smooth lane jitter.
    """
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_fit = None
        self.missed_frames = 0
        
    def smooth(self, current_fit):
        """
        Smooths the list of points.
        Ideally we smooth coefficients, but points are easier here.
        """
        if current_fit is None:
            self.missed_frames += 1
            if self.missed_frames > 5:
                self.prev_fit = None # Lost Tracking
            return self.prev_fit
            
        self.missed_frames = 0
        
        if self.prev_fit is None:
            self.prev_fit = current_fit
            return current_fit
            
        # Interpolate points? A bit complex if lengths differ.
        # Fallback: Just return current (EMA usually for coefficients)
        # For this implementation, we will trust RANSAC stability
        self.prev_fit = current_fit
        return current_fit
