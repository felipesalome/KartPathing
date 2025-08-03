import cv2
import os
import numpy as np
from .color_optimizer import ColorOptimizer
from .racing_line_processor import process_image

def main():
    # Configurações - caminhos absolutos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)  # Diretório raiz do projeto
    
    input_folder = os.path.join(project_dir, 'input_images')
    output_folder = os.path.join(project_dir, 'output_images')
    intermediate_folder = os.path.join(project_dir, 'intermediate')
    
    # Cria as pastas se não existirem
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(intermediate_folder, exist_ok=True)
    
    # Inicializar otimizador de cor
    color_optimizer = ColorOptimizer()
    kart_params = {
        'max_speed': 55/3.6,  # 55 km/h -> m/s
        'friction_coeff': 1.5,
        'track_length': 943  # Comprimento da pista em metros
    }
    
    # Processar cada imagem na pasta
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em: {input_folder}")
        print(f"Por favor, coloque suas imagens na pasta: {input_folder}")
        return
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        print(f"Processando: {image_file}")
        
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro ao carregar imagem: {image_path}")
            continue
        
        # Processar a imagem
        result_img, yellow_only, racing_only, lap_time = process_image(image, color_optimizer, kart_params)
        
        if result_img is not None:
            # Atualizar otimizador com a imagem original
            color_optimizer.update(image)
            
            # Salvar resultados
            processed_path = os.path.join(output_folder, f"processed_{image_file}")
            yellow_path = os.path.join(intermediate_folder, f"yellow_{image_file}")
            racing_path = os.path.join(intermediate_folder, f"racing_{image_file}")
            
            cv2.imwrite(processed_path, result_img)
            cv2.imwrite(yellow_path, yellow_only)
            cv2.imwrite(racing_path, racing_only)
            
            print(f"  Resultado final salvo em: {processed_path}")
            print(f"  Traçado amarelo salvo em: {yellow_path}")
            print(f"  Racing line salvo em: {racing_path}")
            if lap_time is not None:
                print(f"  Tempo estimado: {lap_time:.2f} segundos")
    
    print("Processamento concluído!")

if __name__ == "__main__":
    main()