# Autoresearch Backbones — Soja Research

Data: 2026-03-11 19:33
Tempo total: 247 min

## Histórico de Holdout

| Experimento | Holdout |
|---|---|
| Random split (data leak) | 40.7% |
| Unificação classes EN/PT | 42.0% |
| Split C (folder group) | 47.3% |
| 50/50 campo no treino | 63.8% |
| Segmentação HSV v1 (revertida) | 60.3% |
| Autoresearch EfficientNet (onecycle) | 67.2% |
| Benchmark 3 backbones | 67.2% |

## Melhor Config por Backbone

| Backbone | Params | Val Acc | Val F1 | Holdout Acc | Holdout F1 | Config |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 4.0M | 98.2% | 0.977 | 70.7% | 0.699 | lr=0.0005 d=0.1 onecycle light |
| convnext_base | 87.6M | 92.0% | 0.910 | 63.8% | 0.645 | lr=0.0005 d=0.1 onecycle light |
| maxvit_tiny_tf_224 | 30.4M | 94.7% | 0.917 | 56.9% | 0.535 | lr=0.0002 d=0.4 cosine heavy |

### efficientnet_b0 — Holdout por Classe

| Classe | Corretas | Total | Acc |
|---|---|---|---|
| Ferrugem | 7 | 10 | 70% |
| Mancha-alvo | 8 | 9 | 89% |
| Mosaico | 4 | 7 | 57% |
| Olho-de-rã | 10 | 16 | 62% |
| Oídio | 12 | 16 | 75% |

### convnext_base — Holdout por Classe

| Classe | Corretas | Total | Acc |
|---|---|---|---|
| Ferrugem | 6 | 10 | 60% |
| Mancha-alvo | 6 | 9 | 67% |
| Mosaico | 4 | 7 | 57% |
| Olho-de-rã | 10 | 16 | 62% |
| Oídio | 11 | 16 | 69% |

### maxvit_tiny_tf_224 — Holdout por Classe

| Classe | Corretas | Total | Acc |
|---|---|---|---|
| Ferrugem | 5 | 10 | 50% |
| Mancha-alvo | 7 | 9 | 78% |
| Mosaico | 2 | 7 | 29% |
| Olho-de-rã | 5 | 16 | 31% |
| Oídio | 14 | 16 | 88% |

### efficientnet_b0 — Holdout por Fonte

| Fonte | Corretas | Total | Acc |
|---|---|---|---|
| doencasdeplantas | 28 | 40 | 70% |
| srin | 13 | 18 | 72% |

### convnext_base — Holdout por Fonte

| Fonte | Corretas | Total | Acc |
|---|---|---|---|
| doencasdeplantas | 25 | 40 | 62% |
| srin | 12 | 18 | 67% |

### maxvit_tiny_tf_224 — Holdout por Fonte

| Fonte | Corretas | Total | Acc |
|---|---|---|---|
| doencasdeplantas | 25 | 40 | 62% |
| srin | 8 | 18 | 44% |

## Todos os Experimentos

| # | Backbone | LR | Dropout | Scheduler | Aug | Val Acc | Holdout Acc | Epochs | Tempo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | efficientnet_b0 | 0.0003 | 0.2 | cosine | light | 98.0% | 63.8% | 6 | 901s |
| 2 | efficientnet_b0 | 0.0001 | 0.3 | cosine | medium | 94.9% | 65.5% | 4 | 1371s |
| 3 | efficientnet_b0 | 0.0005 | 0.1 | onecycle | light | 98.2% | 70.7% | 6 | 900s |
| 4 | efficientnet_b0 | 0.001 | 0.2 | cosine | medium | 89.0% | 41.4% | 6 | 833s |
| 5 | efficientnet_b0 | 0.0002 | 0.4 | cosine | heavy | 96.3% | 63.8% | 6 | 894s |
| 6 | convnext_base | 0.0003 | 0.2 | cosine | light | 91.3% | 62.1% | 2 | 900s |
| 7 | convnext_base | 0.0001 | 0.3 | cosine | medium | 94.1% | 60.3% | 2 | 901s |
| 8 | convnext_base | 0.0005 | 0.1 | onecycle | light | 92.0% | 63.8% | 2 | 900s |
| 9 | convnext_base | 0.001 | 0.2 | cosine | medium | 92.6% | 62.1% | 2 | 900s |
| 10 | convnext_base | 0.0002 | 0.4 | cosine | heavy | 94.7% | 58.6% | 2 | 900s |
| 11 | maxvit_tiny_tf_224 | 0.0003 | 0.2 | cosine | light | 97.5% | 53.4% | 3 | 900s |
| 12 | maxvit_tiny_tf_224 | 0.0001 | 0.3 | cosine | medium | 95.8% | 53.4% | 3 | 901s |
| 13 | maxvit_tiny_tf_224 | 0.0005 | 0.1 | onecycle | light | 91.3% | 41.4% | 3 | 900s |
| 14 | maxvit_tiny_tf_224 | 0.001 | 0.2 | cosine | medium | 93.5% | 32.8% | 3 | 900s |
| 15 | maxvit_tiny_tf_224 | 0.0002 | 0.4 | cosine | heavy | 94.7% | 56.9% | 3 | 900s |

## Recomendação

Melhor holdout: **efficientnet_b0** (70.7%)
