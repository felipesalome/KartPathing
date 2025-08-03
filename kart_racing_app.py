import cv2
import numpy as np
import os
import math

# Configurações
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parâmetros ajustáveis
params = {
    # Detecção de cor
    'lower_h': 20,
    'lower_s': 100,
    'lower_v': 100,
    'upper_h': 40,
    'upper_s': 255,
    'upper_v': 255,
    
    # Física do kart
    'max_speed': 55,  # km/h
    'friction': 1.5,  # coeficiente de atrito
    'mass': 170,      # kg (kart + piloto)
    
    # Racing line
    'aggressiveness': 0.7,  # 0.1-1.0 (conservativo-agressivo)
    'smoothness': 0.5,      # 0.1-1.0 (suavização da linha)
    'braking_distance': 15  # metros antes das curvas
}

def detect_yellow_track(image):
    lower = np.array([params['lower_h'], params['lower_s'], params['lower_v']])
    upper = np.array([params['upper_h'], params['upper_s'], params['upper_v']])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    return approx

def calculate_racing_line(contour, track_length_pixels):
    if contour is None or len(contour) < 3:
        return None
    
    points = contour.squeeze()
    n = len(points)
    
    # Calcular escala (pixels por metro)
    track_length_m = 943  # metros (do seu exemplo)
    scale = track_length_pixels / track_length_m
    
    # Converter parâmetros físicos para unidades consistentes
    max_speed_mps = params['max_speed'] / 3.6  # km/h -> m/s
    g = 9.81  # gravidade
    
    # 1. Calcular curvatura em cada ponto
    curvatures = []
    for i in range(n):
        p1 = points[i-2] if i >= 2 else points[n-2+i]
        p2 = points[i-1] if i >= 1 else points[n-1+i]
        p3 = points[i]
        p4 = points[(i+1) % n]
        p5 = points[(i+2) % n]
        
        # Vetores de entrada/saída
        in_vec = p3 - p1
        out_vec = p5 - p3
        
        # Raio de curvatura aproximado
        chord = np.linalg.norm(p5 - p1)
        sagitta = np.linalg.norm(p3 - ((p1 + p5) / 2))
        
        if sagitta > 0 and chord > 0:
            radius = (chord**2) / (8 * sagitta) + sagitta / 2
            curvature = 1 / (radius + 1e-5)  # evitar divisão por zero
        else:
            curvature = 0
            
        curvatures.append(curvature)
    
    # 2. Calcular velocidade máxima em cada ponto
    max_speeds = []
    for curvature in curvatures:
        if curvature > 0:
            # Fórmula física: v_max = sqrt(μ * g * r)
            radius = 1 / curvature
            v_max = math.sqrt(params['friction'] * g * radius)
            v_max = min(max_speed_mps, v_max)
        else:
            v_max = max_speed_mps
        max_speeds.append(v_max)
    
    # 3. Gerar racing line ideal
    racing_line = []
    for i in range(n):
        # Ponto atual
        current = points[i]
        
        # Vetor tangente (direção da pista)
        prev_point = points[i-1] if i > 0 else points[-1]
        next_point = points[(i+1) % n]
        tangent = (next_point - prev_point)
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
        
        # Vetor normal (perpendicular)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Determinar lado ideal para a curva
        if curvatures[i] > 0:
            # Em curvas, mover para o lado externo
            displacement = params['aggressiveness'] * 0.5 * scale
        else:
            # Em retas, manter no centro
            displacement = 0
        
        # Aplicar suavização
        if racing_line:
            prev_rl = racing_line[-1]
            smoothing = params['smoothness'] * 0.1
            displacement = displacement * (1 - smoothing) + (prev_rl - current) * smoothing
        
        # Calcular novo ponto
        new_point = current + displacement * normal
        racing_line.append(new_point)
    
    return np.array(racing_line).reshape(-1, 1, 2)

def draw_racing_line(image, yellow_contour, racing_line):
    result = image.copy()
    
    # Desenhar traçado amarelo
    cv2.drawContours(result, [yellow_contour], -1, (0, 255, 255), 2)
    
    # Desenhar racing line como uma curva suave
    if racing_line is not None:
        pts = racing_line.squeeze()
        
        # Desenhar setas para indicar direção
        for i in range(0, len(pts), 5):
            if i + 1 < len(pts):
                pt1 = tuple(pts[i].astype(int))
                pt2 = tuple(pts[i+1].astype(int))
                cv2.arrowedLine(result, pt1, pt2, (0, 0, 255), 2, tipLength=0.3)
        
        # Desenhar pontos de frenagem nas curvas fechadas
        for i in range(len(pts)):
            if i % 10 == 0:
                cv2.circle(result, tuple(pts[i].astype(int)), 3, (255, 0, 0), -1)
    
    return result

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None, None, None
    
    # Detectar traçado amarelo
    yellow_contour = detect_yellow_track(image)
    
    # Calcular racing line com física realista
    racing_line = None
    if yellow_contour is not None:
        # Calcular comprimento do contorno em pixels
        perimeter = cv2.arcLength(yellow_contour, True)
        racing_line = calculate_racing_line(yellow_contour, perimeter)
    
    # Desenhar resultado
    result = draw_racing_line(image, yellow_contour, racing_line)
    
    return image, yellow_contour, result

def print_instructions():
    """Exibe instruções no terminal"""
    print("\n" + "="*50)
    print("KART RACING LINE OPTIMIZER")
    print("="*50)
    print("Controles:")
    print("  N: Proxima imagem")
    print("  P: Imagem anterior")
    print("  S: Salvar resultado")
    print("  H: Ajustar detecção de cor (HSV)")
    print("  R: Ajustar parâmetros da racing line")
    print("  Q: Sair")
    print("="*50)

def main():
    # Verificar se há imagens na pasta
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em {INPUT_FOLDER}")
        print(f"Coloque suas imagens na pasta '{INPUT_FOLDER}' e execute novamente.")
        return
    
    current_index = 0
    
    while True:
        image_file = image_files[current_index]
        image_path = os.path.join(INPUT_FOLDER, image_file)
        
        original, yellow_contour, result = process_image(image_path)
        
        if original is None:
            print(f"Erro ao processar: {image_file}")
            current_index = (current_index + 1) % len(image_files)
            continue
        
        # Redimensionar para exibição
        display = cv2.resize(result, (1000, 700))
        
        # Mostrar imagem
        cv2.imshow("Kart Racing Line Optimizer", display)
        
        # Exibir instruções no terminal
        print_instructions()
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):  # Próxima imagem
            current_index = (current_index + 1) % len(image_files)
        elif key == ord('p'):  # Imagem anterior
            current_index = (current_index - 1) % len(image_files)
        elif key == ord('s'):  # Salvar
            output_path = os.path.join(OUTPUT_FOLDER, f"opt_{image_file}")
            cv2.imwrite(output_path, result)
            print(f"Resultado salvo em: {output_path}")
        elif key == ord('h'):  # Ajustar detecção de cor
            print("\n=== AJUSTE DE DETECÇÃO DE COR ===")
            params['lower_h'] = int(input("H min (0-179): ") or params['lower_h'])
            params['lower_s'] = int(input("S min (0-255): ") or params['lower_s'])
            params['lower_v'] = int(input("V min (0-255): ") or params['lower_v'])
            params['upper_h'] = int(input("H max (0-179): ") or params['upper_h'])
            params['upper_s'] = int(input("S max (0-255): ") or params['upper_s'])
            params['upper_v'] = int(input("V max (0-255): ") or params['upper_v'])
        elif key == ord('r'):  # Ajustar racing line
            print("\n=== AJUSTE DE RACING LINE ===")
            print("Dica: Para melhorar as curvas, ajuste agressividade e suavidade")
            params['max_speed'] = float(input(f"Velocidade máxima atual: {params['max_speed']} km/h\nNova velocidade (km/h): ") or params['max_speed'])
            params['friction'] = float(input(f"Atrito atual: {params['friction']}\nNovo atrito (1.0-2.0): ") or params['friction'])
            params['aggressiveness'] = float(input(f"Agressividade atual: {params['aggressiveness']:.1f}\nNova agressividade (0.1-1.0): ") or params['aggressiveness'])
            params['smoothness'] = float(input(f"Suavidade atual: {params['smoothness']:.1f}\nNova suavidade (0.1-1.0): ") or params['smoothness'])
            params['braking_distance'] = float(input(f"Distância de frenagem atual: {params['braking_distance']}m\nNova distância (m): ") or params['braking_distance'])
            
            # Dicas para melhorar as curvas
            print("\nDicas para curvas:")
            print(f"- Agressividade: {params['aggressiveness']:.1f} (maior = cortar mais as curvas)")
            print(f"- Suavidade: {params['smoothness']:.1f} (maior = linha mais suave)")
            print(f"- Frenagem: {params['braking_distance']}m (maior = frear mais cedo)")
        elif key == ord('q'):  # Sair
            break
        elif key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()
    print("Aplicativo encerrado!")

if __name__ == "__main__":
    print("OpenCV instalado corretamente. Iniciando aplicativo...")
    main()