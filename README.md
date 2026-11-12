# Detecção de Objetos com YOLOv8 para Pessoas com Deficiência Visual

Sistema em Python que detecta, em tempo real pela webcam, objetos que podem ser obstáculos e **avisa a pessoa por áudio**. Feito para auxiliar pessoas com deficiência visual.

Projeto da Equipe 3 na disciplina de Inteligência Computacional (UFPA), com o professor Aldebaro Klautau.

## Como funciona
1. A webcam captura os frames em tempo real (`yolo.py`).
2. Cada frame é analisado pelo modelo YOLOv8 treinado (`best.pt`).
3. Os objetos detectados são anunciados por voz (`pyttsx3`).

## Dataset
Imagens nossas e de outros usuários, anotadas e rotuladas no [Roboflow](https://universe.roboflow.com/amanda-ow7bi/multiplos-objetos-2-23osn). A versão do dataset está nas pastas `train`, `valid` e `test`.

<!-- TODO: listar as classes de objetos do data.yaml e o número de imagens. -->

## Treino
O modelo foi treinado por **60 épocas** em um notebook do Google Colab (`projeto_IC.ipynb`), a partir de um modelo pré-treinado. O arquivo `best.pt` é o modelo final.

<!-- TODO: adicionar métricas (mAP, precisão, recall) do treino. -->

## Como executar
```bash
git clone https://github.com/a-mand/projeto_IC.git
cd projeto_IC
pip install opencv-python ultralytics numpy pyttsx3
python yolo.py
```
Recomendado rodar dentro de um ambiente virtual Python.

## Estrutura
- `yolo.py`: captura da câmera, inferência e sistema de áudio.
- `best.pt`: modelo treinado.
- `projeto_IC.ipynb`: código e explicação do treino.
- `data.yaml`: caminhos e classes do dataset.
- `train/`, `valid/`, `test/`: dataset da equipe.
- `images_test/`: imagens com as previsões do modelo.

## Tecnologias
Python, YOLOv8 (Ultralytics), OpenCV, pyttsx3, Roboflow, Google Colab.

## Equipe
Amanda Lopes, Filipe Corrêa, Giovanna Cunha e Giovana Nascimento.
