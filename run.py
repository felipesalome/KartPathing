import os
import sys

# Configura caminhos absolutos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório src ao caminho de busca
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from src.main import main

if __name__ == "__main__":
    # Cria a estrutura de pastas necessária
    input_dir = os.path.join(BASE_DIR, 'input_images')
    output_dir = os.path.join(BASE_DIR, 'output_images')
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Pasta de entrada criada: {input_dir}")
        print(f"Por favor, coloque suas imagens nesta pasta e execute novamente.")
    else:
        main()