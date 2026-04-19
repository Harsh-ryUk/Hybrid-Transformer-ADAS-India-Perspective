import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class CurveFitter:
    """
    Fits 2nd Degree Polynomials using RANSAC to ignore outliers (debris).
    Equation: x = ay^2 + by + c
    """
    
    def __init__(self):
        # RANSAC is robust to noise (dirt on road)
        self.model = RANSACRegressor(min_samples=10, residual_threshold=20)
        
    def fit(self, x_pts, y_pts, image_h, image_w):
        if len(x_pts) < 20:
            return None # Not enough points
            
        X = np.array(y_pts).reshape(-1, 1) # Predict x from y
        y = np.array(x_pts)
        
        try:
            # Fit Polynomial (Degree 2)
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly_features.fit_transform(X)
            
            self.model.fit(X_poly, y)
            
            # Generate smooth curve for visualization
            plot_y = np.linspace(image_h * 0.5, image_h, num=50)
            plot_X = poly_features.fit_transform(plot_y.reshape(-1, 1))
            fit_x = self.model.predict(plot_X)
            
            # Filter checks
            # 1. Bounds check
            valid_pts = []
            for px, py in zip(fit_x, plot_y):
                if 0 <= px < image_w:
                    valid_pts.append([int(px), int(py)])
            
            return {
                'points': np.array(valid_pts, dtype=np.int32),
                'coeffs': self.model.estimator_.coef_ if hasattr(self.model, 'estimator_') else []
            }
            
        except Exception as e:
            print(f"Fit Error: {e}")
            return None
