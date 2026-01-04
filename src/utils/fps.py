import time
import collections

class FPSMeter:
    """
    A rolling average FPS counter to provide stable readings for the dashboard.
    """
    def __init__(self, buffer_len=30):
        self._start_time = time.time()
        self._frame_times = collections.deque(maxlen=buffer_len)
        self._prev_time = time.time()

    def tick(self):
        """
        Call this once per frame usually at the end of the loop.
        """
        current_time = time.time()
        # Avoid division by zero by ensuring a small delta
        dt = current_time - self._prev_time
        self._frame_times.append(dt)
        self._prev_time = current_time

    def get_fps(self):
        """
        Returns the rolling average FPS.
        """
        if not self._frame_times:
            return 0.0
        
        # Calculate average time per frame
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        
        if avg_dt == 0:
            return 0.0
            
        return 1.0 / avg_dt

    def get_latency_ms(self):
        """
        Returns the average latency in milliseconds.
        """
        if not self._frame_times:
            return 0.0
        
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return avg_dt * 1000.0
