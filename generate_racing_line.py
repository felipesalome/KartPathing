import cv2
import numpy as np
import os

class RacingLineGenerator:
    def __init__(self):
        self.params = {
            'max_speed': 55/3.6,  # m/s
            'friction_coeff': 1.5,
            'track_length': 943,
            'displacement_factor': 0.5,
            'max_displacement': 20
        }
    
    def generate_racing_line(self, contour):
        if contour is None or len(contour) < 3:
            return None
        
        # Calcula pontos ao longo do contorno
        perimeter = cv2.arcLength(contour, True)
        num_points = 100
        points = []
        
        for i in range(num_points):
            distance = (i / num_points) * perimeter
            point = self.get_point_at_distance(contour, distance)
            points.append(point)
        
        points = np.array(points)
        
        # Gera racing line com deslocamento
        racing_line = []
        for i in range(len(points)):
            prev = points[i-1] if i > 0 else points[-1]
            curr = points[i]
            next_ = points[i+1] if i < len(points)-1 else points[0]
            
            # Vetor de direção
            in_vector = curr - prev
            out_vector = next_ - curr
            
            # Normalizar
            in_len = np.linalg.norm(in_vector)
            out_len = np.linalg.norm(out_vector)
            
            if in_len > 0 and out_len > 0:
                in_vector = in_vector / in_len
                out_vector = out_vector / out_len
                
                # Calcular ângulo
                dot_product = np.dot(in_vector, out_vector)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                # Calcular deslocamento
                displacement = self.params['displacement_factor'] * min(angle, 1.0) * self.params['max_displacement']
                normal = np.array([-in_vector[1], in_vector[0]])
                
                new_point = curr + displacement * normal
            else:
                new_point = curr
                
            racing_line.append(new_point)
        
        return np.array(racing_line).reshape(-1, 1, 2)
    
    def get_point_at_distance(self, contour, distance):
        current_dist = 0
        for i in range(len(contour)-1):
            p1 = contour[i][0]
            p2 = contour[i+1][0]
            segment_length = np.linalg.norm(p2 - p1)
            
            if current_dist + segment_length >= distance:
                ratio = (distance - current_dist) / segment_length
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                return np.array([x, y])
            
            current_dist += segment_length
        
        return contour[-1][0]

def main():
    # Configurações
    yellow_folder = "yellow_tracks"
    output_folder = "racing_lines"
    os.makedirs(output_folder, exist_ok=True)
    
    # Inicializar gerador
    generator = RacingLineGenerator()
    
    # Listar contornos salvos
    contour_files = [f for f in os.listdir(yellow_folder) if f.startswith('contour_') and f.endswith('.npy')]
    
    if not contour_files:
        print("Nenhum contorno encontrado. Execute primeiro o extract_yellow_track.py")
        return
    
    for contour_file in contour_files:
        # Carregar contorno
        contour_path = os.path.join(yellow_folder, contour_file)
        contour = np.load(contour_path, allow_pickle=True)
        
        # Carregar imagem original
        image_name = contour_file.replace('contour_', '').replace('.npy', '')
        image_path = os.path.join("input_images", f"{image_name}.jpg")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Imagem original não encontrada: {image_path}")
            continue
        
        while True:
            # Gerar racing line
            racing_line = generator.generate_racing_line(contour)
            
            # Criar imagem de resultado
            result = image.copy()
            cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)  # Amarelo
            
            if racing_line is not None:
                # Desenhar racing line
                for i in range(1, len(racing_line)):
                    cv2.line(result, 
                             tuple(racing_line[i-1][0].astype(int)), 
                             tuple(racing_line[i][0].astype(int)), 
                             (0, 0, 255), 4)
            
            # Mostrar resultado
            cv2.imshow("Racing Line - Pressione 'a' para aceitar, 'r' para reprocessar", result)
            key = cv2.waitKey(0)
            
            if key == ord('a'):  # Aceitar
                # Salvar resultado
                output_path = os.path.join(output_folder, f"racing_{image_name}.jpg")
                cv2.imwrite(output_path, result)
                print(f"Racing line salva em: {output_path}")
                cv2.destroyAllWindows()
                break
                
            elif key == ord('r'):  # Reprocessar com novos parâmetros
                cv2.destroyAllWindows()
                print("\nAjuste os parâmetros:")
                print("1. Fator de deslocamento (ex: 0.5)")
                print("2. Deslocamento máximo (ex: 20)")
                print("3. Coeficiente de atrito (ex: 1.5)")
                print("0. Voltar ao processamento")
                
                choice = input("Selecione o parâmetro para ajustar (1-3) ou 0 para continuar: ")
                
                if choice == '1':
                    value = float(input("Novo fator de deslocamento: "))
                    generator.params['displacement_factor'] = value
                elif choice == '2':
                    value = float(input("Novo deslocamento máximo: "))
                    generator.params['max_displacement'] = value
                elif choice == '3':
                    value = float(input("Novo coeficiente de atrito: "))
                    generator.params['friction_coeff'] = value
            
            elif key == 27:  # ESC - Sair
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()