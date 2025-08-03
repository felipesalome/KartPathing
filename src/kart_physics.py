import numpy as np
import math

class RacingLineCalculator:
    def __init__(self, max_speed=55/3.6, friction_coeff=1.5):  # max_speed em m/s
        self.max_speed = max_speed
        self.friction_coeff = friction_coeff
        
    def ramer_douglas_peucker(self, points, epsilon):
        """Simplifica uma curva com o algoritmo Ramer-Douglas-Peucker."""
        if len(points) < 3:
            return points
            
        dmax = 0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
                
        if dmax > epsilon:
            rec_results1 = self.ramer_douglas_peucker(points[:index+1], epsilon)
            rec_results2 = self.ramer_douglas_peucker(points[index:], epsilon)
            return np.vstack((rec_results1[:-1], rec_results2))
        else:
            return np.array([points[0], points[end]])
            
    def _perpendicular_distance(self, point, line_start, line_end):
        """Calcula a distância perpendicular de um ponto a uma linha."""
        if np.all(line_start == line_end):
            return np.linalg.norm(point - line_start)
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        
    def calculate_curvatures(self, points):
        """Calcula a curvatura para cada ponto na curva simplificada."""
        curvatures = []
        n = len(points)
        for i in range(n):
            # Pontos anteriores e posteriores
            prev = points[i-1] if i > 0 else points[0]
            curr = points[i]
            next_ = points[i+1] if i < n-1 else points[n-1]
            
            # Vetores
            v1 = curr - prev
            v2 = next_ - curr
            
            # Comprimentos
            len_v1 = np.linalg.norm(v1)
            len_v2 = np.linalg.norm(v2)
            
            if len_v1 > 0 and len_v2 > 0:
                # Ângulo entre os vetores
                cos_theta = np.dot(v1, v2) / (len_v1 * len_v2)
                # Evitar erros numéricos
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
                # Raio de curvatura aproximado
                if theta > 1e-5:
                    curvature = 2 * np.sin(theta) / (len_v1 + len_v2)
                else:
                    curvature = 0
            else:
                curvature = 0
                
            curvatures.append(curvature)
            
        return curvatures
        
    def calculate_optimal_path(self, points):
        if len(points) < 3:
            return points.copy()
            
        # Simplificar o traçado
        simplified = self.ramer_douglas_peucker(points, epsilon=2.0)
        if simplified.shape[0] < 3:
            return simplified
            
        # Calcular curvaturas
        curvatures = self.calculate_curvatures(simplified)
        
        # Gerar racing line
        racing_line = []
        n = len(simplified)
        for i in range(n):
            curvature = curvatures[i]
            # Deslocamento lateral: quanto mais curva, mais para o lado externo
            displacement = self._calculate_displacement(curvature)
            # Ajustar ponto
            if i == 0 or i == n-1:
                new_point = simplified[i]
            else:
                # Direção normal ao segmento
                direction = simplified[i] - simplified[i-1]
                normal = np.array([-direction[1], direction[0]])
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                new_point = simplified[i] + displacement * normal
            racing_line.append(new_point)
            
        return np.array(racing_line)
        
    def _calculate_displacement(self, curvature):
        if curvature < 1e-5:
            return 0
        radius = 1 / curvature
        min_radius = self.max_speed**2 / (self.friction_coeff * 9.8)
        # Se o raio for maior que o mínimo, não precisamos deslocar muito
        if radius > min_radius:
            return 0
        # Deslocamento proporcional à necessidade
        return min_radius - radius  # Apenas um exemplo, pode ser ajustado