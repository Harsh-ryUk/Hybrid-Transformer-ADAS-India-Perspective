"""
Test Suite for ADAS v2.0
Verifies SegFormer, Temporal Fusion, and End-to-End Pipeline.
"""

import unittest
import numpy as np
import cv2
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adas_pipeline_v2 import ADASPipelineV2
from src.utils.temporal_fusion import TemporalConsistencyManager
# Mocking imports if SegFormer model loading is too heavy for CI
# But user wants "Production Grade", so we try real loading (or mock if no GPU)

class TestADASV2(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Setting up V2 Test Suite...")
        # Use CPU for testing to avoid OOM on CI environments
        cls.pipeline = ADASPipelineV2(device='cpu')
        
    def test_pipeline_initialization(self):
        """Verify the pipeline initializes all components."""
        self.assertIsNotNone(self.pipeline.lane_detector)
        self.assertIsNotNone(self.pipeline.damage_detector)
        self.assertIsNotNone(self.pipeline.temporal_manager)
        
    def test_seasonal_detection(self):
        """Verify seasonal condition logic."""
        dark_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cond = self.pipeline.detect_seasonal_condition(dark_frame)
        self.assertEqual(cond, "Night")
        
        bright_frame = np.ones((360, 640, 3), dtype=np.uint8) * 200
        cond = self.pipeline.detect_seasonal_condition(bright_frame)
        self.assertEqual(cond, "Normal")

    def test_temporal_fusion_logic(self):
        """Verify confidence boosting for consistent detections."""
        manager = TemporalConsistencyManager(confidence_boost=0.15)
        
        # Frame 1: Detection at [10, 10, 50, 50]
        det1 = {
            'boxes': [[10, 10, 50, 50]],
            'scores': [0.8],
            'classes': [0]
        }
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        res1, _ = manager.process_frame(frame1, det1)
        
        self.assertEqual(res1['scores'][0], 0.8) # No boost yet
        self.assertEqual(res1['consecutive_frames'][0], 1)
        
        # Frame 2: Same detection (simulate static camera)
        # Using same frame content so Optical Flow is 0, implying detection should be at same spot
        res2, _ = manager.process_frame(frame1, det1)
        
        # Should match and boost
        self.assertTrue(res2['scores'][0] > 0.8)
        self.assertEqual(res2['consecutive_frames'][0], 2)
        
    def test_end_to_end_processing(self):
        """Run full processing on a dummy frame."""
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        viz, metrics = self.pipeline.process_frame(frame)
        
        self.assertEqual(viz.shape, (360, 640, 3))
        self.assertIn('fps', metrics)
        self.assertIn('latency_breakdown', metrics)
        
    def test_latency_targets(self):
        """Check if latency metrics structure exists (Mock performance check)."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        _, metrics = self.pipeline.process_frame(frame)
        breakdown = metrics['latency_breakdown']
        
        print("\nLatency Breakdown (CPU Test):")
        print(breakdown)
        
        self.assertIn('detection', breakdown)
        self.assertIn('lane', breakdown)
        self.assertIn('fusion', breakdown)

if __name__ == '__main__':
    unittest.main()
