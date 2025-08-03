import numpy as np
import cv2

def calculate_centerline(contour, num_points=100):
    """Calcula uma linha central suave para a pista"""
    # Calcula o comprimento total do contorno
    perimeter = cv2.arcLength(contour, True)
    
    # Gera pontos equidistantes ao longo do contorno
    points = []
    for i in range(num_points):
        dist = (i / num_points) * perimeter
        # Obtém o ponto na distância especificada
        point = get_point_at_distance(contour, dist)
        points.append(point)
    
    return np.array(points).reshape((-1, 1, 2))

def get_point_at_distance(contour, distance):
    """Obtém um ponto no contorno a uma certa distância do início"""
    current_dist = 0
    for i in range(len(contour)-1):
        p1 = contour[i][0]
        p2 = contour[i+1][0]
        segment_length = np.linalg.norm(p2 - p1)
        
        if current_dist + segment_length >= distance:
            # Interpola linearmente
            ratio = (distance - current_dist) / segment_length
            x = p1[0] + ratio * (p2[0] - p1[0])
            y = p1[1] + ratio * (p2[1] - p1[1])
            return np.array([x, y])
        
        current_dist += segment_length
    
    # Retorna o último ponto se a distância for maior que o perímetro
    return contour[-1][0]

def generate_racing_line(centerline, max_speed, friction_coeff):
    """Gera a linha de corrida ideal baseada em física"""
    points = centerline.squeeze()
    racing_line = []
    
    # Parâmetros físicos
    g = 9.8  # gravidade
    max_lateral_g = friction_coeff * g
    
    for i in range(len(points)):
        prev_point = points[i-1] if i > 0 else points[-1]
        curr_point = points[i]
        next_point = points[i+1] if i < len(points)-1 else points[0]
        
        # Vetor de direção
        direction = next_point - prev_point
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
            # Vetor normal (perpendicular)
            normal = np.array([-direction[1], direction[0]])
            
            # Calcular curvatura aproximada
            v1 = curr_point - prev_point
            v2 = next_point - curr_point
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                dot = np.dot(v1, v2)
                angle = np.arccos(np.clip(dot, -1.0, 1.0))
                curvature = angle / (np.linalg.norm(v1) + np.linalg.norm(v2))
                
                # Calcular velocidade máxima na curva
                v_max_curve = np.sqrt(max_lateral_g / curvature) if curvature > 0 else max_speed
                v_target = min(max_speed, v_max_curve)
                
                # Calcular deslocamento lateral
                displacement_factor = (v_target / max_speed) * 0.5
                displacement = displacement_factor * normal * 20
            else:
                displacement = np.array([0, 0])
        else:
            displacement = np.array([0, 0])
            
        racing_point = curr_point + displacement
        racing_line.append(racing_point)
    
    return np.array(racing_line).reshape((-1, 1, 2))