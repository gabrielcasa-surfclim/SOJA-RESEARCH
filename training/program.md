# Programa do Agente Autônomo — Classificação de Doenças em Soja

> Baseado no autoresearch do Karpathy, adaptado para visão computacional agrícola.
> Objetivo: maximizar acurácia no validation set via experimentação automatizada.

## Regras Absolutas

1. **NUNCA modifique `prepare.py`** — ele é fixo e garante comparação justa entre experimentos.
2. **NUNCA PARE** — rode experimentos indefinidamente até ser interrompido.
3. **Modifique APENAS a seção HYPERPARAMETERS** em `train.py` (entre os marcadores `██`).
4. **Logue tudo** — cada experimento deve aparecer em `results.tsv`.
5. **Uma mudança por vez** — modifique UM hiperparâmetro por experimento para isolar o efeito.

## Setup Inicial

```bash
cd training
cat prepare.py   # Entenda o dataset, classes, e métricas
cat train.py     # Entenda a estrutura e os hiperparâmetros editáveis
```

## Loop Principal

```
LOOP INFINITO:
    1. Leia results.tsv para entender o histórico de experimentos
    2. Escolha UMA mudança de hiperparâmetro (baseado no histórico)
    3. Edite train.py (APENAS a seção HYPERPARAMETERS)
    4. git add train.py && git commit -m "exp: [descrição da mudança]"
    5. python train.py
    6. Leia a saída — anote acurácia, precision, recall, f1
    7. SE melhorou (✅ NOVO RECORDE):
         → git add -A && git commit -m "best: [acurácia]% - [o que mudou]"
         → Essa config vira o novo baseline
    8. SE não melhorou (❌):
         → Reverta train.py ao estado anterior do melhor resultado
         → git checkout train.py  (volta pro último best)
         → git commit -m "revert: [o que tentou] não melhorou"
    9. Próximo experimento → volte ao passo 1
```

## Estratégia de Exploração

### Fase 1 — Baseline (primeiros 5 experimentos)
Rode o train.py como está (defaults). Isso estabelece o baseline.
Depois teste variações óbvias:
- `FREEZE_BACKBONE = True` (treinar só o head primeiro)
- `LEARNING_RATE = 0.0003` (LR menor)
- `AUGMENTATION_LEVEL = "heavy"`
- `BATCH_SIZE = 32`

### Fase 2 — Refinamento de LR (experimentos 6-15)
O learning rate é o hiperparâmetro mais impactante. Teste:
- LRs: 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01
- Schedulers: cosine, onecycle, step
- Combine o melhor LR com o melhor scheduler

### Fase 3 — Augmentation e Regularização (experimentos 16-30)
Com o melhor LR fixado:
- AUGMENTATION_LEVEL: none → light → medium → heavy
- DROPOUT: 0.0, 0.1, 0.2, 0.3, 0.5
- LABEL_SMOOTHING: 0.0, 0.05, 0.1, 0.15
- WEIGHT_DECAY: 0, 1e-5, 1e-4, 1e-3

### Fase 4 — Arquitetura (experimentos 31-50)
Com os melhores hiperparâmetros de treino:
- MODEL: efficientnet_b0, efficientnet_b1, mobilenet_v3_large, resnet18, resnet34
- IMAGE_SIZE: 224, 256, 320 (modelos maiores podem usar imagens maiores)
- Teste FREEZE_BACKBONE=True → fine-tune gradual

### Fase 5 — Combos e Polimento (experimentos 51+)
- Combine as melhores descobertas de cada fase
- Tente combinações não-óbvias
- Fine-tune: ajuste fino de LR em torno do melhor valor (ex: se 0.001 foi melhor, tente 0.0008 e 0.0012)
- Revise a confusion matrix para identificar classes problemáticas

## Hiperparâmetros Editáveis

| Variável | Valores Válidos | Default |
|---|---|---|
| MODEL | efficientnet_b0, efficientnet_b1, mobilenet_v3_small, mobilenet_v3_large, resnet18, resnet34 | efficientnet_b0 |
| LEARNING_RATE | 0.0001 — 0.01 | 0.001 |
| BATCH_SIZE | 8, 16, 32 | 16 |
| EPOCHS | 10 — 100 (limitado pelo budget de 5min) | 30 |
| IMAGE_SIZE | 224, 256, 320 | 224 |
| DROPOUT | 0.0 — 0.5 | 0.2 |
| OPTIMIZER | adam, adamw, sgd | adam |
| SCHEDULER | cosine, step, onecycle, none | cosine |
| FREEZE_BACKBONE | True, False | False |
| AUGMENTATION_LEVEL | none, light, medium, heavy | medium |
| WEIGHT_DECAY | 0 — 0.01 | 1e-4 |
| LABEL_SMOOTHING | 0.0 — 0.2 | 0.1 |

## Dicas

- **Confusion matrix é sua amiga**: se Ferrugem e Mancha-alvo estão se confundindo, tente augmentation mais agressivo ou imagens maiores.
- **Se overfit** (train acc >> val acc): aumente dropout, augmentation, weight_decay, ou congele backbone.
- **Se underfit** (train acc baixa): aumente LR, descongele backbone, reduza regularização.
- **Budget de 5 min**: modelos maiores (B1, ResNet34) fazem menos epochs — compense com LR maior ou OneCycleLR.
- **MPS quirks**: batch_size 32 pode dar OOM no M4 16GB com imagens 320px. Se falhar, reduza batch ou image_size.
- **Se train.py crashar**: leia o erro, corrija APENAS a seção de hiperparâmetros, e tente de novo.

## Formato do results.tsv

O arquivo é gerado automaticamente pelo train.py. Colunas:
```
timestamp  model  lr  batch_size  epochs_completed  image_size  dropout  optimizer  scheduler  freeze_backbone  augmentation  weight_decay  label_smoothing  accuracy  precision  recall  f1  loss  elapsed_seconds  improved
```

Use este arquivo como memória. Antes de cada experimento, leia-o para decidir o que tentar.
