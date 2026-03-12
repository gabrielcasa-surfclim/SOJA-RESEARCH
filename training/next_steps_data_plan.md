# Plano do Próximo Ciclo — Foco em Dados de Campo

**Data**: 2026-03-11
**Objetivo**: Subir holdout de 70.7% para 85%+ focando em dados, não em modelo.

---

## 1. Diagnóstico: Por que 70.7% e não 90%?

O gap de 27.5pp entre validação (98.2%) e holdout (70.7%) tem uma causa clara:

- **99.5% do treino é laboratorial** (fundo controlado, iluminação uniforme)
- **100% do holdout é campo** (fundo complexo, luz variável, ângulos diversos)
- O modelo aprendeu features de laboratório, não de campo

O backbone (EfficientNet-B0) e os hiperparâmetros já estão otimizados. **Mais dados de campo é o único caminho para melhoria significativa.**

---

## 2. Classes que Precisam de Mais Fotos

### Prioridade ALTA (recall < 60%)
| Classe | Holdout | Treino | Problema |
|--|--|--|--|
| **Mosaico** | 57% (4/7) | 272 imgs | Menor classe do dataset (3.3%). Confunde com Olho-de-rã |
| **Olho-de-rã** | 62% (10/16) | 1255 imgs | Volume ok, mas confunde com Ferrugem e Mancha-alvo |

### Prioridade MÉDIA (recall 70-75%)
| Classe | Holdout | Treino | Problema |
|--|--|--|--|
| **Ferrugem** | 70% (7/10) | 3174 imgs | Volume alto mas poucos exemplos de campo |
| **Oídio** | 75% (12/16) | 1107 imgs | Performance razoável, pode melhorar com campo |

### Prioridade BAIXA
| Classe | Holdout | Treino | Problema |
|--|--|--|--|
| **Mancha-alvo** | 89% (8/9) | 1696 imgs | Já performa bem |
| **Saudável** | — | 808 imgs | Sem holdout, mas 98.5% das imgs são PlantVillage |

---

## 3. Fontes Fracas

### Fonte mais fraca: dados de campo no treino
| Fonte | Treino | % do Total | Tipo |
|--|--|--|--|
| digipathos | 4069 | 48.9% | Lab (folhas sobre papel) |
| asdid | 3399 | 40.9% | Lab (fundo controlado) |
| plantvillage | 800 | 9.6% | Lab (fundo uniforme) |
| **doencasdeplantas** | **34** | **0.4%** | **Campo** |
| **srin** | **10** | **0.1%** | **Campo** |

**44 imagens de campo no treino para um holdout de 58 fotos de campo.** O modelo vê quase zero exemplos do domínio que precisa generalizar.

---

## 4. Meta de Coleta: Novas Imagens de Campo

### Meta mínima (viabilizar 80%+)
| Classe | Alvo | Por quê |
|--|--|--|
| Mosaico | **50+ fotos** | Classe mais subrepresentada, pior recall |
| Olho-de-rã | **40+ fotos** | Segundo pior recall, confusões frequentes |
| Ferrugem | **30+ fotos** | Muitas de lab, poucas de campo |
| Oídio | **30+ fotos** | Melhorar generalização campo |
| Mancha-alvo | **20+ fotos** | Já boa, manter cobertura |
| Saudável | **30+ fotos** | Precisa de referência de campo |

**Total mínimo: ~200 fotos de campo**

### Meta ideal (viabilizar 85%+)
- 100+ fotos por classe = **500+ fotos de campo**
- Fotografar com celular em campo real
- Variar: ângulo, distância, iluminação (sol, sombra, nublado)
- Incluir folhas em diferentes estágios da doença (inicial → avançado)

### Como coletar
1. **Tio agrônomo**: principal fonte. Pedir fotos durante visitas a lavouras
2. **WhatsApp**: criar grupo para ele enviar fotos tagueadas
3. **Protocolo simples**: "Tira 3 fotos de cada folha: close-up, meia distância, folha inteira na planta"
4. **Metadados**: pedir que informe doença, localização, data, estágio

---

## 5. Priorizar Mosaico e Olho-de-rã

### Mosaico (prioridade #1)
- **Problema**: 272 imgs de treino (todas lab), 57% holdout
- **Confusão principal**: Mosaico → Olho-de-rã (3/7 erros)
- **Ação**: Coletar 50+ fotos de mosaico em campo
- **Variações**: padrão leve vs severo, folhas jovens vs velhas
- **Data augmentation**: Mixup entre Mosaico e classes confusoras pode ajudar

### Olho-de-rã (prioridade #2)
- **Problema**: 62% holdout, confunde com Ferrugem (3 erros) e Oídio (2 erros)
- **Ação**: Coletar 40+ fotos focando em:
  - Diferentes tamanhos de lesão
  - Contraste com Ferrugem (lesões mais escuras vs avermelhadas)
  - Diferentes estágios (centro cinza definido vs indefinido)

---

## 6. Incorporar Novas Fotos sem Contaminar Holdout

### Regra de Ouro
**NUNCA adicionar fotos ao holdout ou mover fotos entre splits depois de ver resultados.** O holdout atual (58 fotos) é sagrado — é o único indicador honesto de performance.

### Processo para Novas Fotos

```
1. Novas fotos chegam → data/images/campo_v2_[classe]/
2. Segmentar com segment_leaf.py → data/images_segmented/campo_v2_[classe]/
3. Adicionar TODAS ao treino (NÃO ao holdout)
4. Atualizar data/splits/train.csv e val.csv (80/20 dentro das novas)
5. Re-treinar com mesma config vencedora
6. Avaliar no MESMO holdout de 58 fotos
7. Se melhorou → commit; se piorou → investigar
```

### Script sugerido: `add_field_photos.py`
- Recebe pasta com novas fotos
- Valida formato e resolução mínima
- Atribui classe (manual ou semi-automático)
- Faz split 80/20 DENTRO das novas fotos
- Appenda em train.csv e val.csv
- Roda segmentação nas novas
- NÃO toca no holdout.csv

---

## 7. Holdout Externo v2

### Por que criar um novo holdout?
O holdout atual (58 fotos) é pequeno e já foi "observado" várias vezes. Embora nunca treinemos nele, o risco de **overfitting indireto** (otimizar decisões baseadas no holdout) existe.

### Proposta: Holdout v2
| Aspecto | Holdout v1 (atual) | Holdout v2 (proposta) |
|--|--|--|
| Tamanho | 58 fotos | 100+ fotos |
| Fontes | doencasdeplantas, srin | **Novas fotos do tio** |
| Status | Usado como benchmark | **Guardado em cofre, avaliado 1x/mês** |
| Classes | 5 (sem Saudável) | 6 (com Saudável) |

### Como montar
1. Pedir ao tio **20+ fotos por classe** que ele tenha **certeza absoluta** do diagnóstico
2. Salvar em `data/splits/holdout_v2.csv`
3. **NÃO usar para decisões de hiperparâmetros**
4. Avaliar no máximo 1x por mês, em "dia de benchmark"
5. Se holdout v2 divergir muito do v1, investigar a causa

---

## 8. Cronograma Sugerido

| Semana | Ação | Meta |
|--|--|--|
| 1 | Pedir fotos ao tio (Mosaico + Olho-de-rã) | 50+ fotos |
| 2 | Processar fotos, adicionar ao treino | Re-treinar |
| 3 | Coletar mais (Ferrugem + Oídio + Saudável) | +100 fotos |
| 4 | Montar holdout v2, re-treinar | 80%+ holdout v1 |

---

## 9. Resumo Executivo

- **Modelo está otimizado** — EfficientNet-B0 com onecycle é o teto para este dataset
- **Dados são o gargalo** — 99.5% lab vs 100% campo no holdout
- **Prioridade #1**: Mosaico (57% recall, 272 imgs de treino)
- **Prioridade #2**: Olho-de-rã (62% recall, confusões frequentes)
- **Meta**: 200+ fotos de campo → 80%+ holdout
- **Não contaminar holdout** — novas fotos só no treino/val
