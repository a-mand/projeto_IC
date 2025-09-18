# Projeto de Detecção de Objetos com YOLOv8
Este repositório contém a implementação em Python do projeto de detecção de objeto, utilizando o Visual Studio Code (VS Code), Notebook Colab Google e bibliotecas como `OpenCV`, `Ultralytics`, `pyttsx3` e `Roboflow`.
## Visão Geral do Projeto
O objetivo principal é fornecer uma ferramenta que alerte pessoas com deficiência visual. A ferramenta deve ser capaz de identificar objetos que possam ser impencílios e alertar o indivíduo através de um sistema de áudio.
### Explicação do Treino:
O `projeto_IC.ipybn` possui a explicação do treino do modelo, porém para mais detalhes será explicado nosso caminho do dataset ao projeto final:
#### Passo a Passo:
1. Dataset

Para iniciarmos nosso projeto é necessário um conjunto de imagens para treinarmos nosso modelo, para isso utilizamos o Roboflow. O Roboflow é uma ferramenta online onde os usuários disponibilizam imagens gratuitamente. Segue link do nosso dataset online:

    https://universe.roboflow.com/amanda-ow7bi/multiplos-objetos-2-23osn

Reunimos imagens nossas e dos usuários para o treino e teste. Ao realizar todas as anotações nas imagens e definir as classes e labels, salvamos nossa versão do dataset e baixarmos ele ulizando o código pronto do Roboflow (código no colab). O dataset está disponível neste repositório.

2. Treino

Em um ambiente preparado (Colab Notebook) rodamos 60 épocas do nosso modelo. Ao terminar o treinamento fazemos a extração do arquivo best.py no caminho indicado no final do treino.

O ``arquivo best.pt`` é a nossa inteligência, onde nosso modelo treinado está e o responsável pela nossa detectação.

3. Código Python

O código ``yolo.py`` é onde capturamos os frames em tempo real através da webcam, enviamos para serem analisados pelo ``best.pt`` e finalmente ele nos retorna a previsão da detecção.

## Como usar a Análise em Tempo Real
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/a-mand/projeto_IC.git)
    cd projeto_IC
    ```
2.  **Instale as dependências para o modelo:**
    ```bash
    pip install opencv-python pyttsx3 numpy ultralytics time threading
    ```
3.  **Execute o programa principal:**
    ```bash
    python yolo.py
    ```
    Preferível rodá-lo dentro de um ambiente python.
## Explicações dos Resultados


## Estrutura do Repositório
* `images_test`: Contém imagens com as previsões do modelo.
*  `train`, `test` e `valid`: Pastas contendo o dataset da equipe.
* `Relatório/`: Pasta contendo o relatório acadêmico e suas figuras geradas.
    * `Relatório/Figuras/`: Contém as imagens dos gráficos e saídas do console para as questões resolvidas.
* `data.yaml`: Arquivo com os caminhos e classes, usando para guiar o modelo para o dataset no treinamento.
* `yolo.py`: Arquivo responsável pela conexão com a câmera e pelo sistema de áudio.
* `projeto_IC.ipybn`: Arquivo onde está disponivel o código do treino com o nosso dataset do modelo pré-treinado do Roboflow
## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.
## Autor
AMANDA LOPES - 202207040043

FILIPE CORRÊA - 202006840020

GIOVANNA NASCIMENTO - 202206840015

GIOVANA CUNHA - 202206840039
