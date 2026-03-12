# Benchmark de Backbones — Soja Research

Data: 2026-03-11 11:08

## Histórico de Experimentos

| Experimento | Holdout | Nota |
|---|---|---|
| Random split (data leak) | 40.7% | leak entre train/val |
| Unificação classes EN/PT | 42.0% | Target Spot→Mancha-alvo |
| Split C (folder group) | 47.3% | sem leak, 112 imgs holdout |
| 50/50 campo no treino | 63.8% | 58 imgs holdout |
| Segmentação HSV (revertida) | 60.3% | perdeu tecido doente |

## Comparativo de Backbones

| Modelo | Params | Val Acc | Val F1 | Holdout Acc | Holdout F1 | Epochs | Tempo | Batch |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 5.3M | 97.2% | 0.965 | 63.8% | — | 3 | 300s | 32 |
| convnext_base | 87.6M | 94.8% | 0.922 | 56.9% | 0.567 | 0 | 481s | 16 |
| maxvit_tiny_tf_224 | 30.4M | 95.7% | 0.946 | 50.0% | 0.477 | 1 | 481s | 16 |

### convnext_base — Holdout por Classe

| Classe | Corretas | Total | Acc |
|---|---|---|---|
| Ferrugem | 6 | 10 | 60% |
| Mancha-alvo | 6 | 9 | 67% |
| Mosaico | 7 | 7 | 100% |
| Olho-de-rã | 5 | 16 | 31% |
| Oídio | 9 | 16 | 56% |

### maxvit_tiny_tf_224 — Holdout por Classe

| Classe | Corretas | Total | Acc |
|---|---|---|---|
| Ferrugem | 4 | 10 | 40% |
| Mancha-alvo | 7 | 9 | 78% |
| Mosaico | 2 | 7 | 29% |
| Olho-de-rã | 6 | 16 | 38% |
| Oídio | 10 | 16 | 62% |

### convnext_base — Holdout por Fonte

| Fonte | Corretas | Total | Acc |
|---|---|---|---|
| doencasdeplantas | 21 | 40 | 52% |
| srin | 12 | 18 | 67% |

### maxvit_tiny_tf_224 — Holdout por Fonte

| Fonte | Corretas | Total | Acc |
|---|---|---|---|
| doencasdeplantas | 21 | 40 | 52% |
| srin | 8 | 18 | 44% |

## Recomendação

Melhor holdout: **efficientnet_b0** (63.8%)

## Config

- LR: 0.0003
- Image size: 224
- Dropout: 0.2
- Optimizer: adamw
- Scheduler: cosine
- Augmentation: light
- Label smoothing: 0.1
- Budget: 480s por modelo
