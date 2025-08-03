import cv2
import numpy as np
import os
from src.image_processor import detect_yellow_track
from src.racing_line_processor import generate_racing_line, draw_racing_line

# Configurações
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parâmetros ajustáveis
params = {
    'lower_h': 20,
    'lower_s': 200,
    'lower_v': 200,
    'upper_h': 40,
    'upper_s': 255,
    'upper_v': 255,
    'displacement': 0.3
}

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    
    # Detectar traçado amarelo
    lower = np.array([params['lower_h'], params['lower_s'], params['lower_v']])
    upper = np.array([params['upper_h'], params['upper_s'], params['upper_v']])
    yellow_contour = detect_yellow_track(image, lower, upper)
    
    # Gerar racing line
    racing_line = None
    if yellow_contour is not None:
        racing_line = generate_racing_line(yellow_contour, params['displacement'])
    
    # Desenhar resultado
    result = image.copy()
    if yellow_contour is not None:
        cv2.drawContours(result, [yellow_contour], -1, (0, 255, 255), 2)
    
    if racing_line is not None:
        result = draw_racing_line(result, yellow_contour, racing_line)
    
    return image, yellow_contour, result

def main():
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em {INPUT_FOLDER}")
        return
    
    current_index = 0
    
    while True:
        image_file = image_files[current_index]
        image_path = os.path.join(INPUT_FOLDER, image_file)
        
        original, yellow_contour, result = process_image(image_path)
        
        if original is None:
            print(f"Erro ao carregar: {image_file}")
            current_index = (current_index + 1) % len(image_files)
            continue
        
        # Mostrar imagem
        display = cv2.resize(result, (1000, 700))
        cv2.imshow("Kart Racing Line Optimizer", display)
        
        # Instruções
        print("\nControles:")
        print("  N/P: Próxima/Imagem anterior")
        print("  S: Salvar resultado")
        print("  H: Ajustar parâmetros HSV")
        print("  D: Ajustar deslocamento")
        print("  Q: Sair")
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):  # Próxima imagem
            current_index = (current_index + 1) % len(image_files)
        elif key == ord('p'):  # Imagem anterior
            current_index = (current_index - 1) % len(image_files)
        elif key == ord('s'):  # Salvar
            output_path = os.path.join(OUTPUT_FOLDER, f"opt_{image_file}")
            cv2.imwrite(output_path, result)
            print(f"Resultado salvo em: {output_path}")
        elif key == ord('h'):  # Ajustar HSV
            print("\nAjuste os parâmetros HSV (0-255):")
            params['lower_h'] = int(input("Lower H: ") or params['lower_h'])
            params['lower_s'] = int(input("Lower S: ") or params['lower_s'])
            params['lower_v'] = int(input("Lower V: ") or params['lower_v'])
            params['upper_h'] = int(input("Upper H: ") or params['upper_h'])
            params['upper_s'] = int(input("Upper S: ") or params['upper_s'])
            params['upper_v'] = int(input("Upper V: ") or params['upper_v'])
        elif key == ord('d'):  # Ajustar deslocamento
            new_disp = float(input("Novo fator de deslocamento (0.1-1.0): ") or params['displacement'])
            params['displacement'] = max(0.1, min(1.0, new_disp))
        elif key == ord('q'):  # Sair
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()