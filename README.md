# Projeto de reconhecimento de gestos

Este projeto usa webcam + OpenCV + MediaPipe para reconhecer gestos com a mao em tempo real.

## Recursos da versao atual

- reconhecimento de `OK`, `NO`, `JOINHA`, `PAZ`, `PARE` e `PUNHO`
- historico de frames para estabilizar a deteccao
- confianca, FPS, mao detectada e contagem de dedos na tela
- painel lateral com status e atalhos
- coleta de amostras para criar um dataset proprio
- treino de classificador local com `scikit-learn`
- modo desafio com pontuacao
- logs de sessao em CSV

## Como executar

1. Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Instale as dependencias:

```powershell
pip install -r requirements.txt
```

3. Rode a aplicacao:

```powershell
.venv\Scripts\python main.py
```

Na primeira execucao, o projeto baixa automaticamente o modelo oficial da MediaPipe e verifica a integridade dele.

## Atalhos

- `Q`: sair
- `1` a `6`: trocar o rotulo de coleta
- `C`: salvar amostra do gesto atual no dataset local
- `T`: treinar um classificador local com o dataset salvo
- `G`: iniciar ou pausar o desafio

## Treinar por linha de comando

Se preferir, tambem da para treinar fora da interface:

```powershell
.venv\Scripts\python train_model.py
```

## Dataset local

As amostras salvas pela interface vao para:

- `data/gesture_samples.csv`

O modelo treinado localmente vai para:

- `artifacts/gesture_classifier.joblib`
- `artifacts/gesture_labels.json`
