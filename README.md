# 🚗 YOLOv4 Transfer Learning para Detecção de Pessoas e Carros

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![YOLOv4](https://img.shields.io/badge/YOLOv4-Darknet-orange?style=for-the-badge)
![CUDA](https://img.shields.io/badge/NVIDIA%20CUDA-12.4-brightgreen?style=for-the-badge&logo=nvidia)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-lightgrey?style=for-the-badge&logo=googlecolab)

## 📄 Descrição do Projeto

Este repositório documenta o processo de fine-tuning do modelo de detecção de objetos **YOLOv4** utilizando a técnica de **Transfer Learning**. O objetivo foi especializar um modelo pré-treinado no robusto dataset **MS COCO** para detectar com alta precisão apenas duas classes de interesse: **Pessoa 🚶** e **Carro 🚗**.

Todo o pipeline, desde a configuração do ambiente até o treinamento, foi executado em ambiente **Google Colab**, aproveitando a aceleração de hardware com uma GPU NVIDIA Tesla T4.

## 🛠️ Tecnologias e Frameworks Utilizados

*   **Modelo:** YOLOv4
*   **Framework Core:** Darknet (fork de [AlexeyAB](https://github.com/AlexeyAB/darknet))
*   **Técnica Principal:** Transfer Learning
*   **Dataset Base:** MS COCO 2017
*   **Pesos Pré-treinados:** `yolov4.conv.137`
*   **Linguagem:** Python 3
*   **Ambiente:** Google Colab
*   **Hardware:** GPU NVIDIA Tesla T4
*   **Bibliotecas e Ferramentas:** `nvidia-smi`, `sed`, `wget`, `OpenCV`, `CUDA`, `cuDNN`

## ⚙️ Pipeline do Projeto

O desenvolvimento seguiu um pipeline estruturado, totalmente contido no notebook `Transfer_Learning_com_YOLOv4_e_COCO_no_Google_Colab_LMB.ipynb`. As principais etapas são:

### 1. 🖥️ Configuração do Ambiente
   - Verificação e alocação de uma GPU NVIDIA no ambiente Colab.
   - Montagem do Google Drive para persistência de dados, datasets e pesos do modelo.
   - Clonagem do repositório Darknet de AlexeyAB, que é a implementação de referência para o treinamento.

### 2. 🔧 Compilação do Darknet
   - Modificação do `Makefile` do Darknet para habilitar o suporte essencial à aceleração de hardware e processamento de imagem:
     - `GPU=1`
     - `CUDNN=1`
     - `CUDNN_HALF=1` (para utilizar Tensor Cores da GPU T4)
     - `OPENCV=1`
   - Compilação do framework a partir do código-fonte.

### 3. 🖼️ Preparação dos Dados
   - Download do dataset MS COCO 2017 (imagens de treino, validação e anotações).
   - Execução de um script Python customizado para processar as anotações:
     - O script lê os arquivos de anotação no formato COCO (JSON).
     - **Filtra** todas as anotações, mantendo apenas as pertencentes às classes `person` e `car`.
     - **Converte** o formato das bounding boxes de `[x_min, y_min, width, height]` (COCO) para o formato normalizado `[center_x, center_y, width, height]` exigido pelo YOLO.
     - Gera os arquivos `.txt` de anotações correspondentes a cada imagem, contendo apenas as classes de interesse.

### 4. 📝 Configuração do Modelo para Transfer Learning
   - Criação dos arquivos de metadados (`.names` e `.data`) para o treinamento customizado.
   - Download dos pesos convolucionais pré-treinados (`yolov4.conv.137`), que contêm o conhecimento extraído das 137 primeiras camadas do modelo treinado no ImageNet.
   - Adaptação do arquivo de configuração `yolov4-custom.cfg`:
     - Ajuste do `batch` e `subdivisions` para otimizar o uso da memória da GPU T4.
     - Cálculo e ajuste de `max_batches` para o novo número de classes (Regra geral: `classes * 2000`).
     - Alteração do parâmetro `classes` para **2** nas três camadas YOLO.
     - Recálculo do número de `filters` nas camadas convolucionais que precedem cada camada YOLO, usando a fórmula `(classes + 5) * 3`.

### 5. 🚀 Treinamento
   - Início do processo de treinamento com o comando `!./darknet detector train ...`.
   - Utilização da flag `-map` para calcular e exibir o Mean Average Precision (mAP) periodicamente, permitindo o monitoramento da performance do modelo durante o treinamento.

## 📈 Resultados (A ser preenchido)

