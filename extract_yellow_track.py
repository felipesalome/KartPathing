import cv2
import numpy as np
import os

class YellowTrackExtractor:
    def __init__(self):
        self.params = {
            'lower_hsv': [20, 100, 100],
            'upper_hsv': [40, 255, 255],
            'morph_size': 7,
            'epsilon_factor': 0.001
        }
    
    def detect_yellow_track(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(self.params['lower_hsv'])
        upper = np.array(self.params['upper_hsv'])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Operações morfológicas
        kernel = np.ones((self.params['morph_size'], self.params['morph_size']), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        main_contour = max(contours, key=cv2.contourArea)
        epsilon = self.params['epsilon_factor'] * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        return approx

    def adjust_parameter(self, param_name, value):
        if param_name in self.params:
            self.params[param_name] = value
            print(f"Parâmetro {param_name} atualizado para: {value}")

def main():
    # Configurações
    input_folder = "input_images"
    output_folder = "yellow_tracks"
    os.makedirs(output_folder, exist_ok=True)
    
    # Inicializar extrator
    extractor = YellowTrackExtractor()
    
    # Listar imagens
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Nenhuma imagem encontrada na pasta input_images.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        while True:
            # Processar imagem
            yellow_track = extractor.detect_yellow_track(image)
            
            # Criar imagem de resultado
            result = image.copy()
            if yellow_track is not None:
                cv2.drawContours(result, [yellow_track], -1, (0, 255, 255), 3)
            
            # Mostrar resultado
            cv2.imshow("Traçado Amarelo - Pressione 'a' para aceitar, 'r' para reprocessar", result)
            key = cv2.waitKey(0)
            
            if key == ord('a'):  # Aceitar
                # Salvar resultado
                output_path = os.path.join(output_folder, f"yellow_{image_file}")
                cv2.imwrite(output_path, result)
                
                # Salvar contorno para uso posterior
                contour_path = os.path.join(output_folder, f"contour_{os.path.splitext(image_file)[0]}.npy")
                np.save(contour_path, yellow_track)
                
                print(f"Traçado salvo em: {output_path}")
                cv2.destroyAllWindows()
                break
                
            elif key == ord('r'):  # Reprocessar com novos parâmetros
                cv2.destroyAllWindows()
                print("\nAjuste os parâmetros:")
                print("1. Limite inferior HSV (ex: 20,100,100)")
                print("2. Limite superior HSV (ex: 40,255,255)")
                print("3. Tamanho do kernel morfológico (ex: 7)")
                print("4. Fator epsilon (ex: 0.001)")
                print("0. Voltar ao processamento")
                
                choice = input("Selecione o parâmetro para ajustar (1-4) ou 0 para continuar: ")
                
                if choice == '1':
                    values = input("Digite novos valores para lower_hsv (H,S,V): ").split(',')
                    if len(values) == 3:
                        extractor.adjust_parameter('lower_hsv', [int(v) for v in values])
                elif choice == '2':
                    values = input("Digite novos valores para upper_hsv (H,S,V): ").split(',')
                    if len(values) == 3:
                        extractor.adjust_parameter('upper_hsv', [int(v) for v in values])
                elif choice == '3':
                    value = int(input("Novo tamanho do kernel: "))
                    extractor.adjust_parameter('morph_size', value)
                elif choice == '4':
                    value = float(input("Novo fator epsilon: "))
                    extractor.adjust_parameter('epsilon_factor', value)
            
            elif key == 27:  # ESC - Sair
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()