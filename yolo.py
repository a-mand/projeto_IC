import cv2
from ultralytics import YOLO
import numpy as np
import time
import threading
import pyttsx3

# 1. Inicializa Text-to-Speech (TTS)
engine = pyttsx3.init()

# 2. Procura e configura uma voz em Português (pt-BR)
voices = engine.getProperty('voices')
voz_pt_br_encontrada = False
for voice in voices:
    if 'brazil' in voice.name.lower() or 'portuguese' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        print(f"Voz em Português encontrada e selecionada: {voice.name}")
        voz_pt_br_encontrada = True
        break
if not voz_pt_br_encontrada:
    print("Aviso: Nenhuma voz em Português (Brasil) foi encontrada. Usando a voz padrão.")

# 3. Configura a velocidade da fala
engine.setProperty('rate', 180)

# 4. Dicionário para rastrear o último alerta de áudio dos objetos NA TELA
objetos_rastreados_audio = {}

COOLDOWN_SEGUNDOS = 4 # Intervalo para repetir o alerta do mesmo objeto

# 5. Função para falar o texto em uma thread
def falar_alerta(texto):
    engine.say(texto)
    engine.runAndWait()

# Carregar o modelo treinado
model = YOLO('best.pt')

# Definir a URL RTSP
rtsp_url = "rtsp://admin:P5EB3N89@192.168.137.159:554/cam/realmonitor?channel=1&subtype=0"

# Criar o objeto de captura de vídeo
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Erro: Não foi possível conectar ao stream. Verifique a URL.")
else:
    print("Conexão estabelecida. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do stream ou erro de leitura.")
            break

        frame = cv2.resize(frame, (640, 480))
        largura_frame = frame.shape[1]

        results = model(frame)
                
        # 1. Crie um conjunto com as classes de todos os objetos detectados NESTE frame
        classes_detectadas_no_frame = set()
        
        # Dicionário para guardar a posição do primeiro objeto de cada classe no frame
        posicoes_neste_frame = {}

        for box in results[0].boxes:
            confianca = box.conf[0]

            if confianca >= 0.6:
                id_classe = int(box.cls[0])
                nome_classe = model.names[id_classe]
                
                # Adiciona a classe ao conjunto de objetos do frame atual
                classes_detectadas_no_frame.add(nome_classe)
                
                # Guarda a posição do primeiro objeto de cada classe (para usar no alerta)
                if nome_classe not in posicoes_neste_frame:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    posicoes_neste_frame[nome_classe] = coords

                # Desenha a caixa e o rótulo no frame (parte visual)
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                rotulo = f'{nome_classe} {confianca:.2f}'
                cv2.putText(frame, rotulo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 2. Agora, processe a lógica de áudio com base nos objetos visíveis
        agora = time.time()
        for nome_classe in classes_detectadas_no_frame:
            # Se o objeto é novo ou o cooldown passou, fale o alerta
            if nome_classe not in objetos_rastreados_audio or agora - objetos_rastreados_audio[nome_classe] > COOLDOWN_SEGUNDOS:
                
                # Pega as coordenadas guardadas para este objeto
                x1, y1, x2, y2 = posicoes_neste_frame[nome_classe]
                centro_x_objeto = (x1 + x2) / 2
                
                posicao = ""
                if centro_x_objeto < largura_frame / 3:
                    posicao = "à sua esquerda"
                elif centro_x_objeto > largura_frame * (2/3):
                    posicao = "à sua direita"
                else:
                    posicao = "à sua frente"

                mensagem = f"{nome_classe} {posicao}"
                thread_audio = threading.Thread(target=falar_alerta, args=(mensagem,))
                thread_audio.start()
                
                # Atualiza o tempo do último alerta para este objeto
                objetos_rastreados_audio[nome_classe] = agora

        # 3. Limpe da memória os objetos que NÃO estão mais na tela
        objetos_para_remover = [nome for nome in objetos_rastreados_audio if nome not in classes_detectadas_no_frame]
        for nome in objetos_para_remover:
            del objetos_rastreados_audio[nome]
            print(f"Objeto '{nome}' saiu de vista, resetando alerta.")
            
        cv2.imshow("Detecção para Acessibilidade Visual", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()