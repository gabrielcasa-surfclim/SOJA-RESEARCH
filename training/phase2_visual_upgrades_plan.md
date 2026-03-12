# Phase 2 — Visual Upgrades Plan

**Objetivo**: Subir holdout de 70.7% para 85%+ sem coletar novas fotos.
**Princípio**: Cada fase é um experimento isolado. Se não melhorar, reverte.

---

## Fase 0 — Baseline Congelado (ATUAL)

**Status**: COMPLETO

- EfficientNet-B0, segmentação seletiva v2
- Holdout: 70.7% (41/58)
- Config: lr=0.0005, dropout=0.1, onecycle, light aug, AdamW
- Checkpoint: `best_model_production.pth`
- Arquivo de controle: `phase2_baseline_locked.json`
- Resultados: `phase2_results.csv`

---

## Fase 1 — Augmentation (CONCLUÍDA — Baseline Light confirmado ótimo)

**Status**: CONCLUÍDA — Nenhum augmentation superou o baseline.

**Experimentos realizados**:
- **B1: RandAugment moderado** → REJEITADO (holdout 68.97%, -1.72pp)
  - ColorJitter com hue/saturation matou Ferrugem (70%→50%) — cores das pústulas são sinal diagnóstico
- **B2: RandAugment sem color jitter** → REJEITADO (holdout 65.52%, -5.17pp)
  - Recuperou Ferrugem (80%) mas destruiu Mancha-alvo (89%→78%), Olho-de-rã (62%→50%), Mosaico (57%→43%)
  - Mesmo augmentation geométrico puro (rotação, blur, erasing) piora generalização

**Experimentos cancelados**:
- B3 (CutMix): CANCELADO — augmentation não é o caminho
- B4 (class-weighted aug): CANCELADO — mesma razão

**Conclusão**: Augmentation "light" (apenas flip horizontal + rotação ±10° + brightness ±10%) é o ótimo para este dataset. Augmentation mais agressivo causa overfitting nas transformações em vez de melhorar generalização.

---

## Fases 2-6 — CANCELADAS

**Razão**: O gargalo não é modelo, augmentation, balanceamento ou resolução. O gargalo é **domain shift**:
- **99.5% das imagens de treino** são de laboratório (fundo controlado, iluminação uniforme)
- **100% do holdout** são fotos de campo (fundo natural, iluminação variável)
- Nenhuma técnica de treinamento vai resolver essa diferença fundamental

**O que resolve**: Coletar 200+ fotos de campo (via tio Ricardo) e re-treinar com dados do domínio correto.

---

## Regras do Processo

1. **Cada experimento roda 1x** com a mesma config documentada
2. **Mesmo holdout** (58 fotos) para todos — nunca tocar
3. **Se melhorou** → salvar checkpoint, logar em `phase2_results.csv`, commit
4. **Se piorou** → logar mesmo assim, seguir para próximo experimento
5. **Melhor de cada fase** vira baseline da fase seguinte
6. **Seed=42** para reprodutibilidade (onde aplicável)
7. **Budget**: 15 min por experimento (mesmo do autoresearch)
8. **Backbones**: Só EfficientNet-B0 (já provado superior)

---

## Próximo Passo

**Coletar dados de campo** — ver `next_steps_data_plan.md` para detalhes.

Prioridade de coleta:
1. Mosaico (57% holdout, apenas 272 imgs treino)
2. Olho-de-rã (62% holdout)
3. Ferrugem (70% holdout, confunde com Mancha-alvo)
4. Oídio (75% holdout)
5. Mancha-alvo (89% holdout — já OK)

*Fase 2 encerrada em 2026-03-12. Baseline light (B0) permanece como modelo de produção.*
