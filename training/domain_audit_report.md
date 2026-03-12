# Domain Audit — Fontes de Treino

**Data**: 2026-03-11
**Total**: 10385 imagens (train+val)


## Resumo por Fonte

| Fonte | Imgs | Resolução Avg | Brightness | Sharpness | BG Removed | Domínio |
|--|--|--|--|--|--|--|
| asdid | 4248 | 5058x3473 | 125 (±23) | 1159 (±1594) | 7.5% | domain_clean |
| digipathos | 5083 | 320x301 | 110 (±20) | 371 (±640) | 0.3% | domain_clean |
| doencasdeplantas | 40 | 2678x1869 | 135 (±18) | 2191 (±2344) | 1.8% | domain_varied |
| plantvillage | 1000 | 256x256 | 148 (±16) | 6001 (±3549) | 32.3% | domain_clean |
| srin | 14 | 1300x994 | 128 (±10) | 766 (±961) | 0.6% | domain_varied |

## Classes por Fonte

| Fonte | Ferrugem | Mancha-alvo | Mosaico | Olho-de-rã | Oídio | Saudável |
|--|--|--|--|--|--|--|
| asdid | 1627 | 1081 | 0 | 1540 | 0 | 0 |
| digipathos | 2330 | 1028 | 333 | 15 | 1368 | 9 |
| doencasdeplantas | 9 | 9 | 0 | 8 | 14 | 0 |
| plantvillage | 0 | 0 | 0 | 0 | 0 | 1000 |
| srin | 0 | 0 | 6 | 7 | 1 | 0 |

## Domain Groups

### domain_clean (10331 imgs)
Imagens laboratoriais com fundo controlado, iluminação uniforme, resolução consistente

Fontes: asdid, digipathos, plantvillage

### domain_varied (54 imgs)
Imagens de campo ou com variação natural de iluminação, resolução e fundo

Fontes: doencasdeplantas, srin


## Domain Mismatch

- Treino: 99.5% clean, 0.5% varied
- Holdout: 0% clean, 100% varied
- **Gap**: modelo treina em 99% lab, testa em 100% campo
