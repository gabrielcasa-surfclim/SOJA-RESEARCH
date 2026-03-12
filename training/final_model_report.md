# Relatório Final — Modelo de Classificação de Doenças em Soja

**Data**: 2026-03-11
**Modelo Vencedor**: EfficientNet-B0 (segmentado)
**Holdout**: 70.7% (58 fotos de campo)

---

## 1. Evolução do Projeto

| # | Experimento | Holdout | Delta | Insight |
|--|--|--|--|--|
| 1 | Random split (data leak) | 40.7% | — | Splits aleatórios vazavam dados entre treino/teste |
| 2 | Unificação de classes EN/PT | 42.0% | +1.3pp | Target Spot→Mancha-alvo, Frogeye→Olho-de-rã |
| 3 | Split pareado (folder group) | 47.3% | +5.3pp | Splits por pasta eliminaram leak residual |
| 4 | Fotos de campo no treino | 63.8% | +16.5pp | Incluir 50% das fotos doencasdeplantas/srin no treino |
| 5 | Segmentação HSV v1 (revertida) | 60.3% | -3.5pp | HSV removia tecido doente junto com fundo |
| 6 | Autoresearch original (onecycle) | 67.2% | +6.9pp | 5 configs × EfficientNet-B0, onecycle venceu |
| 7 | Segmentação v2 + Autoresearch | **70.7%** | +3.5pp | Hull+GrabCut preserva folha inteira |
| — | ConvNeXt-Base (benchmark) | 63.8% | — | 88M params, só 2 epochs no budget |
| — | MaxViT-Tiny (benchmark) | 56.9% | — | 30M params, 3 epochs, pior generalização |

**Ganho total**: 40.7% → 70.7% = **+30.0pp em ~48h de iteração**

---

## 2. Modelo Vencedor

### Backbone
- **Arquitetura**: EfficientNet-B0 (torchvision, pretrained ImageNet)
- **Parâmetros**: 4.0M (vs 87.6M ConvNeXt, 30.4M MaxViT)
- **Eficiência**: 17.6 holdout%/M-param (24x mais eficiente que ConvNeXt)

### Configuração Vencedora
| Hiperparâmetro | Valor |
|--|--|
| Learning Rate | 0.0005 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Augmentation | light |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.05 |
| Batch Size | 32 |
| Image Size | 224 |
| Epochs | 6 (budget 15 min) |
| Imagens | Segmentadas (v2: hull+GrabCut) |

### Métricas
| Métrica | Validação | Holdout |
|--|--|--|
| Accuracy | 98.2% | 70.7% |
| F1 Macro | 0.977 | 0.699 |
| Precision | 0.972 | 0.714 |
| Recall | 0.983 | 0.707 |

---

## 3. Desempenho por Classe (Holdout)

| Classe | Corretas | Total | Accuracy | Observação |
|--|--|--|--|--|
| Mancha-alvo | 8 | 9 | **89%** | Melhor classe |
| Oídio | 12 | 16 | **75%** | Boa performance |
| Ferrugem | 7 | 10 | **70%** | Aceitável |
| Olho-de-rã | 10 | 16 | **62%** | Confunde com outras manchas |
| Mosaico | 4 | 7 | **57%** | Pior classe — poucos exemplos de treino |
| Saudável | — | 0 | — | Não presente no holdout |

### Gargalos por Classe
- **Mosaico (57%)**: Apenas 272 imagens de treino (3.3% do dataset). Subrepresentada.
- **Olho-de-rã (62%)**: 1255 imagens de treino, mas confunde com Mancha-alvo e Ferrugem.
- **Ferrugem (70%)**: 3174 imagens, mas holdout tem poucas amostras (10).

---

## 4. Desempenho por Fonte (Holdout)

| Fonte | Corretas | Total | Accuracy |
|--|--|--|--|
| doencasdeplantas | 28 | 40 | **70%** |
| srin | 13 | 18 | **72%** |

Fontes de campo (não-laboratoriais) mostram performance similar — o modelo generaliza razoavelmente para dados novos.

---

## 5. Matriz de Confusão (Holdout — Modelo Vencedor)

```
              Ferrugem  Mancha-alvo  Mosaico  Olho-de-rã  Oídio
Ferrugem           7          3        0          0        0
Mancha-alvo        1          8        0          0        0
Mosaico            0          0        4          3        0
Olho-de-rã         3          1        0         10        2
Oídio              1          1        1          1       12
```

### Confusões Mais Frequentes
1. **Ferrugem → Mancha-alvo** (3 erros): lesões jovens de ferrugem confundem
2. **Mosaico → Olho-de-rã** (3 erros): padrões de coloração similares
3. **Olho-de-rã → Ferrugem** (3 erros): manchas circulares confundem
4. **Olho-de-rã → Oídio** (2 erros): folhas claras confundem

---

## 6. Comparativo de Backbones (Segmentado)

| Backbone | Params | Val Acc | Holdout Acc | Holdout F1 | Epochs | Eficiência |
|--|--|--|--|--|--|--|
| **EfficientNet-B0** | **4.0M** | **98.2%** | **70.7%** | **0.699** | 6 | 17.6 |
| ConvNeXt-Base | 87.6M | 92.0% | 63.8% | 0.645 | 2 | 0.73 |
| MaxViT-Tiny | 30.4M | 94.7% | 56.9% | 0.535 | 3 | 1.87 |

ConvNeXt e MaxViT sofrem por completar poucos epochs no budget de 15 min. EfficientNet-B0 é o melhor custo-benefício para este dataset e hardware (Mac Mini M4).

---

## 7. Dataset Atual

| Split | Total | Ferrugem | Mancha-alvo | Mosaico | Olho-de-rã | Oídio | Saudável |
|--|--|--|--|--|--|--|--|
| Treino | 8312 | 3174 | 1696 | 272 | 1255 | 1107 | 808 |
| Validação | 2073 | 792 | 422 | 67 | 315 | 276 | 201 |
| Holdout | 58 | 10 | 9 | 7 | 16 | 16 | 0 |

### Fontes no Treino
| Fonte | Imagens | Tipo |
|--|--|--|
| digipathos | 4069 | Laboratorial |
| asdid | 3399 | Laboratorial |
| plantvillage | 800 | Laboratorial |
| doencasdeplantas | 34 | Campo |
| srin | 10 | Campo |

**Problema central**: 99.5% do treino é laboratorial, mas o holdout é 100% campo.

---

## 8. Segmentação v2

- **Método**: HSV color detection → convex hull (filled) → GrabCut refinement
- **Impacto**: +3.5pp no holdout (67.2% → 70.7%)
- **Efetiva em**: 18.1% das imagens (principalmente PlantVillage e ASDID com fundo)
- **Recomendação**: Cenário C (seletivo) — segmentar onde remove fundo, manter original no resto

---

## 9. Checkpoints

| Arquivo | Descrição |
|--|--|
| `best_model_production.pth` | Modelo oficial para produção |
| `best_efficientnet_b0.pth` | Mesmo checkpoint (backup) |
| `best_convnext_base.pth` | Melhor ConvNeXt (referência) |
| `best_maxvit_tiny.pth` | Melhor MaxViT (referência) |

---

## 10. Conclusão

O EfficientNet-B0 com segmentação v2 e OneCycleLR é o melhor modelo atual. O gap val→holdout (98.2% → 70.7%) indica que **o gargalo não é o modelo, é o dataset**. O próximo ciclo deve focar em coletar mais fotos de campo, especialmente de Mosaico e Olho-de-rã.
