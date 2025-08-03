import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ColorOptimizer:
    def __init__(self, n_clusters=3):
        self.model = MiniBatchKMeans(n_clusters=n_clusters)
        self.samples = []
        self.limits = [np.array([20, 200, 200]), np.array([40, 255, 255])]  # Valores padrão
        
    def update(self, new_image):
        hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.limits[0], self.limits[1])
        yellow_pixels = hsv[mask > 0]
        
        if len(yellow_pixels) > 0:
            self.samples.extend(yellow_pixels)
            
            if len(self.samples) > 1000:
                self.model.fit(np.array(self.samples))
                self._update_limits()
    
    def _update_limits(self):
        centers = self.model.cluster_centers_
        h_center = np.median(centers[:,0])
        s_min = np.percentile(centers[:,1], 10)
        v_min = np.percentile(centers[:,2], 10)
        
        self.limits = [
            np.array([max(0, h_center-10), max(0, s_min), max(0, v_min)]),
            np.array([min(180, h_center+10), 255, 255])
        ]
    
    def get_limits(self):
        return self.limits