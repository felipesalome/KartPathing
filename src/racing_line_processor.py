import cv2
import numpy as np

def generate_racing_line(contour, displacement_factor=0.3):
    if contour is None or len(contour) < 3:
        return None
    
    points = contour.squeeze()
    racing_line = []
    
    for i in range(len(points)):
        prev = points[i-1] if i > 0 else points[-1]
        curr = points[i]
        next_ = points[i+1] if i < len(points)-1 else points[0]
        
        # Vetores de direção
        in_vec = curr - prev
        out_vec = next_ - curr
        
        # Normalizar
        if np.linalg.norm(in_vec) > 0:
            in_vec = in_vec / np.linalg.norm(in_vec)
        if np.linalg.norm(out_vec) > 0:
            out_vec = out_vec / np.linalg.norm(out_vec)
        
        # Calcular normal
        normal = np.array([-in_vec[1], in_vec[0]])
        
        # Calcular curvatura
        dot = np.dot(in_vec, out_vec)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        curvature = angle / (np.linalg.norm(in_vec) + np.linalg.norm(out_vec) + 1e-5)
        
        # Aplicar deslocamento
        displacement = displacement_factor * curvature * 50
        new_point = curr + displacement * normal
        racing_line.append(new_point)
    
    return np.array(racing_line).reshape(-1, 1, 2)

def draw_racing_line(image, yellow_contour, racing_line):
    result = image.copy()
    cv2.drawContours(result, [yellow_contour], -1, (0, 255, 255), 2)
    
    if racing_line is not None:
        # Desenhar linha suave
        for i in range(1, len(racing_line)):
            pt1 = tuple(racing_line[i-1][0].astype(int))
            pt2 = tuple(racing_line[i][0].astype(int))
            cv2.line(result, pt1, pt2, (0, 0, 255), 3)
    
    return result