Este projeto utiliza a biblioteca OpenCV para detectar, rastrear e calcular a velocidade de veículos em um vídeo com imagens de uma rodovia. 
Além disso, é possível contar os veículos que cruzam linhas específicas definidas no quadro do vídeo.

## Como executar o código
Para executar o código basta bastar as bibliotecas no arquivo `requirements.txt` com o comando:

```
pip install -r requirements.txt
```

Após isso basta executar o arquivo `main.py`

```
python3 main.py
```

## Funções principais
Todas as funções utilizadas na aplicação se encontram no arquivo `main.py`.

#### preprocess_frame
- Realiza subtração de background.
- Aplica filtros para redução de ruído.
- Converte em iamgens binárias.

#### get_momentum_centroid
Calcula o centróide de um contorno

#### detect_and_track_cars
- Detecta veículos com base em contornos.
- Calcula o centróide de cada veículo.
- Utiliza um histórico de contornos para calcular a distância entre pontos.

#### count_cars
- Conta o número de bounding boxes passando por uma linha.
