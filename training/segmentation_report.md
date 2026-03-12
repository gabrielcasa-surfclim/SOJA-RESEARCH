# Relatório de Segmentação v2

**Total de imagens analisadas**: 10443


## Distribuição por Categoria

| Categoria | Count | % |
|--|--|--|
| minor_change | 5301 | 50.8% |
| bg_removed | 1892 | 18.1% |
| significant_change | 1886 | 18.1% |
| identical | 1364 | 13.1% |

## Por Fonte

| Fonte | Total | Avg Diff | Avg BG Removed | BG Count |
|--|--|--|--|--|
| asdid | 4248 | 17.9 | 7.6% | 856 |
| digipathos | 5083 | 1.8 | 0.3% | 33 |
| doencasdeplantas | 80 | 5.7 | 1.6% | 5 |
| plantvillage | 1000 | 62.1 | 32.3% | 994 |
| srin | 32 | 4.1 | 1.6% | 4 |

## Por Classe

| Classe | Total | Avg Diff | Avg BG Removed |
|--|--|--|--|
| Ferrugem | 3976 | 6.1 | 2.1% |
| Mancha-alvo | 2127 | 18.7 | 8.1% |
| Mosaico | 346 | 1.8 | 0.3% |
| Olho-de-rã | 1586 | 12.2 | 4.8% |
| Oídio | 1399 | 1.4 | 0.1% |
| Saudável | 1009 | 61.6 | 32.0% |

## Cenários

- **A**: Original — todas 10443 imagens sem segmentação
- **B**: Segmentado completo — todas 10443 imagens segmentadas
- **C**: Seletivo — 3778 segmentadas + 6665 originais

## Recomendação

**Cenário C**: Segmentação efetiva em 18.1% das imagens (63.8% sem mudança). Usar cenário seletivo (Cenário C) para maximizar benefício.

Resultado empírico: original 67.2% → segmentado 70.7% (+3.5pp holdout)
