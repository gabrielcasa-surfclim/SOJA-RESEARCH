# SOJA RESEARCH — Plataforma de Pesquisa e Diagnóstico de Doenças em Soja

> Ferramenta de IA que identifica doenças em soja a partir de fotos de folhas, combinada com um agregador de pesquisa científica.

## Visão Geral

Plataforma web que agrega pesquisa científica, organiza banco de imagens, e futuramente treina modelos de visão computacional para diagnóstico autônomo de doenças em soja e milho. O agrônomo tira foto da folha no campo e recebe diagnóstico em segundos.

**Colaborador-chave**: Tio agrônomo que atua em campo (visitando lavouras) e consultoria. Valida o modelo com experiência prática e tem rede de contatos para distribuição.

**Problema**: Doenças de soja representam 8-25% de perda anual de produtividade. No Brasil (maior produtor mundial de soja) = bilhões de reais em perdas evitáveis. Diagnóstico visual depende de experiência pessoal e pode ser impreciso no estágio inicial.

---

## Arquitetura em 3 Etapas

```
ETAPA 1 (atual) — Plataforma Web de Pesquisa
├── Agregador de artigos científicos (PubMed, CrossRef, Semantic Scholar)
├── Banco de imagens organizado por doença
├── Fichas técnicas por patologia
├── Upload e classificação de imagens
└── Galeria visual pra consulta no campo

ETAPA 2 (com dados) — Treinamento Autônomo (estilo autoresearch/Karpathy)
├── Dataset: imagens coletadas + fotos do tio
├── Agente IA itera no train.py autonomamente
├── ~100 experimentos/noite no Mac Mini M4
├── EfficientNet-B0 com transfer learning
└── Métrica: acurácia no validation set

ETAPA 3 (com modelo) — App de Campo
├── React Native + Expo (reusa stack do Eli)
├── Foto → diagnóstico em segundos
├── Funciona offline (modelo embarcado)
├── Histórico com GPS (mapa de ocorrências)
└── Feedback loop: tio corrige → modelo melhora
```

---

## Stack Técnica

| Componente | Tecnologia | Motivo |
|--|--|--|
| Frontend Web | React + TypeScript + Vite + Tailwind | Rápido, moderno, dark mode |
| Backend | Supabase (PostgreSQL + Auth + Storage) | Mesmo do Eli, já dominado |
| Busca de Papers | PubMed API + CrossRef API + Semantic Scholar API | Grátis, oficiais, sem scraping |
| Resumo de Papers | DeepSeek-R1 8B (Ollama, local) | Custo zero |
| Treinamento ML | PyTorch + MPS (Apple Silicon) | GPU do M4 nativa |
| Modelo Base | EfficientNet-B0 (transfer learning) | Melhor custo-benefício pra mobile |
| App Mobile (futuro) | React Native + Expo | Reusa do Eli |
| Conversão Mobile | CoreML (iOS) + TFLite (Android) | Modelo embarcado |

**Ambiente**: Mac Mini M4 16GB, Claude Code, DeepSeek-R1 local, Qwen3 4B local

---

## Estrutura do Projeto

```
soja-research/
├── CLAUDE.md                        # Este arquivo — contexto completo
├── src/
│   ├── components/
│   │   ├── Layout.tsx               # Layout com sidebar de navegação
│   │   ├── DiseaseCard.tsx          # Card de doença (foto, nome, stats)
│   │   ├── ImageGallery.tsx         # Galeria de imagens com filtros
│   │   ├── ArticleCard.tsx          # Card de artigo científico
│   │   ├── SearchBar.tsx            # Barra de busca universal
│   │   ├── UploadZone.tsx           # Drag & drop de imagens
│   │   └── StatsCard.tsx            # Cards de estatísticas
│   ├── pages/
│   │   ├── Dashboard.tsx            # Visão geral: doenças + stats + progresso
│   │   ├── DiseasePage.tsx          # Detalhe de 1 doença (info + galeria + artigos)
│   │   ├── Research.tsx             # Busca de artigos científicos
│   │   ├── Gallery.tsx              # Banco de imagens completo
│   │   ├── Upload.tsx               # Upload e classificação de fotos
│   │   └── Training.tsx             # Futuro: status do treinamento autônomo
│   ├── data/
│   │   └── diseases.ts              # 7 doenças-alvo com metadados completos
│   ├── lib/
│   │   ├── supabase.ts              # Cliente Supabase
│   │   └── api/
│   │       ├── pubmed.ts            # PubMed E-utilities API
│   │       ├── crossref.ts          # CrossRef API
│   │       ├── semanticScholar.ts   # Semantic Scholar API
│   │       └── index.ts             # Busca combinada em todas as fontes
│   └── types/
│       └── index.ts                 # Tipos TypeScript completos
├── scripts/
│   ├── seed-diseases.ts             # Popular banco com 7 doenças
│   ├── fetch-articles.ts            # Buscar artigos via APIs
│   └── fetch-datasets.ts           # Baixar datasets públicos (PlantVillage, Kaggle)
├── supabase/
│   └── schema.sql                   # Schema completo com indexes e triggers
├── training/                        # Futuro: autoresearch adaptado
│   ├── prepare.py                   # Dataset loading + evaluation (fixo)
│   ├── train.py                     # Modelo + loop de treino (agente modifica)
│   └── program.md                   # Instruções pro agente autônomo
└── package.json
```

---

## 7 Doenças-Alvo

| # | Doença | Nome Científico | Sintoma Visual | Severidade |
|--|--|--|--|--|
| 1 | Ferrugem Asiática | *Phakopsora pachyrhizi* | Pústulas marrom-avermelhadas na parte inferior da folha | Alta |
| 2 | Mancha-olho-de-rã | *Cercospora sojina* | Lesões circulares com centro cinza e borda marrom | Média |
| 3 | Oídio | *Erysiphe diffusa* | Pó branco/cinza na superfície das folhas | Média |
| 4 | Antracnose | *Colletotrichum truncatum* | Manchas escuras irregulares, necrose | Alta |
| 5 | Mosaico (vírus) | BPMV / SMV | Padrão de mosaico verde claro/escuro | Média |
| 6 | Mancha-alvo | *Corynespora cassiicola* | Lesões com anéis concêntricos (formato de alvo) | Alta |
| 7 | Folha Saudável | — | Referência de controle | — |

### Confusões Comuns (o modelo precisa aprender)
- Ferrugem × Mancha-alvo (quando jovem)
- Oídio × Resíduo de calda de fungicida
- Mancha-olho-de-rã × Mancha-alvo (fases similares)
- Dano mecânico × Doença (vento, granizo)

### Edge Cases Importantes
- Múltiplas doenças na mesma folha
- Estágio inicial vs avançado (performance diferente)
- Pragas vs doenças (inseto vs fungo)
- Saudável com dano mecânico

**Estratégia**: começar com 3-4 classes no MVP (Ferrugem + Saudável + Oídio + Mancha-alvo), expandir depois.

---

## Banco de Dados (Supabase)

### 6 Tabelas

```sql
-- Doenças-alvo
diseases (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    scientific_name VARCHAR(200),
    symptoms TEXT[],
    severity VARCHAR(20),           -- 'baixa' | 'média' | 'alta'
    conditions TEXT,                 -- condições favoráveis
    management TEXT[],               -- recomendações de manejo
    fungicides TEXT[],               -- fungicidas recomendados
    created_at TIMESTAMP
)

-- Imagens classificadas por doença
disease_images (
    id UUID PRIMARY KEY,
    disease_id UUID REFERENCES diseases(id),
    url TEXT NOT NULL,
    thumbnail_url TEXT,
    source VARCHAR(50),              -- 'plantvillage' | 'kaggle' | 'field' | 'upload'
    metadata JSONB,                  -- {stage, weather, location, validated_by}
    validated BOOLEAN DEFAULT false,
    created_at TIMESTAMP
)

-- Artigos científicos
articles (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT[],
    abstract TEXT,
    doi VARCHAR(100) UNIQUE,
    source VARCHAR(50),              -- 'pubmed' | 'crossref' | 'semanticscholar'
    url TEXT,
    published_date DATE,
    methods TEXT[],                   -- métodos de ML usados
    accuracy FLOAT,                  -- acurácia reportada
    dataset_used TEXT,               -- dataset usado no paper
    summary TEXT,                    -- resumo gerado pelo DeepSeek
    created_at TIMESTAMP
)

-- Relação artigos ↔ doenças (muitos-para-muitos)
article_disease (
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    disease_id UUID REFERENCES diseases(id) ON DELETE CASCADE,
    relevance_score FLOAT,           -- 0-1
    PRIMARY KEY (article_id, disease_id)
)

-- Feedback do tio/especialista
image_feedback (
    id UUID PRIMARY KEY,
    image_id UUID REFERENCES disease_images(id),
    original_disease_id UUID REFERENCES diseases(id),
    suggested_disease_id UUID REFERENCES diseases(id),
    confidence VARCHAR(20),          -- 'certeza' | 'suspeita' | 'não sei'
    notes TEXT,
    gps_lat FLOAT,
    gps_lng FLOAT,
    weather VARCHAR(20),             -- 'sol' | 'nublado' | 'chuva'
    created_at TIMESTAMP
)

-- Experimentos de treinamento (futuro)
training_experiments (
    id UUID PRIMARY KEY,
    model_name VARCHAR(100),         -- 'efficientnet-b0' | 'mobilenet-v3'
    config JSONB,                    -- {lr, batch_size, epochs, augmentation}
    metrics JSONB,                   -- {accuracy, precision, recall, f1, confusion_matrix}
    status VARCHAR(20),              -- 'pending' | 'running' | 'completed' | 'failed'
    commit_hash VARCHAR(7),
    val_accuracy FLOAT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP
)
```

### Indexes
```sql
CREATE INDEX idx_images_disease ON disease_images(disease_id);
CREATE INDEX idx_images_validated ON disease_images(validated);
CREATE INDEX idx_images_source ON disease_images(source);
CREATE INDEX idx_articles_doi ON articles(doi);
CREATE INDEX idx_articles_source ON articles(source);
CREATE INDEX idx_articles_title ON articles USING GIN (title gin_trgm_ops);
CREATE INDEX idx_experiments_status ON training_experiments(status);
CREATE INDEX idx_experiments_accuracy ON training_experiments(val_accuracy DESC);
```

---

## APIs de Busca de Artigos

### PubMed (NCBI E-utilities) — Grátis, sem auth
```
Busca: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=20&retmode=json
Detalhe: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json
Abstract: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id}&retmode=xml
Rate limit: 3 req/s (sem API key), 10 req/s (com API key gratuita)
```

### CrossRef — Grátis, email no header
```
Busca: https://api.crossref.org/works?query={query}&rows=20
Rate limit: 50 req/s (com email no header Mailto:)
Filtros: filter=from-pub-date:{year}
```

### Semantic Scholar — Grátis, API key gratuita
```
Busca: https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=20&fields=title,abstract,authors,year,citationCount
Rate limit: 100 req/dia (sem key), 5000/dia (com key gratuita)
```

### Termos de Busca Otimizados por Doença
```
Ferrugem: "soybean rust" OR "Phakopsora pachyrhizi" AND "detection" AND "deep learning"
Oídio: "soybean powdery mildew" OR "Erysiphe diffusa" AND "image classification"
Mancha-alvo: "target spot soybean" OR "Corynespora cassiicola" AND "computer vision"
Geral: "soybean disease detection" AND ("CNN" OR "transfer learning" OR "EfficientNet")
```

---

## Datasets Públicos

| Dataset | Classes | Imagens | Fonte |
|--|--|--|--|
| PlantVillage | 3 (soja) + 4 (milho) | ~6.400 | Kaggle/GitHub |
| Mendeley Soybean | 8 classes | ~9.500 | Mendeley Data |
| Southern Illinois Univ. | 5 níveis dano + saudável | 2.930 | Repositório acadêmico |
| Kaggle Soybean Diseases | 6+ classes | Variável | Kaggle |

### Estratégia de Coleta
1. Começar com PlantVillage + Kaggle (mais acessíveis)
2. Complementar com Mendeley pra categorias específicas
3. Enriquecer com fotos do campo tiradas pelo tio
4. Data augmentation agressivo pra expandir dataset

---

## Estado da Arte (referência)

| Modelo | Acurácia | Dataset | Paper |
|--|--|--|--|
| DenseNet201 | 96.8% | ~9.500 img, 8 classes | ScienceDirect 2022 |
| RANet | 98.49% | Olho-de-rã + Phyllosticta | Yu et al., 2022 |
| VGG16 (transfer) | 99.35% | 6.958 img, 6 classes | 2025 |
| YOLOv8-DML | 96.9% mAP50 | Múltiplas doenças | Frontiers 2025 |
| EfficientNetB4 | 94.29% | Ferrugem, 3 culturas | Shahoveisi et al., 2023 |

**Insight**: modelos treinados com imagens de folhas AINDA NA PLANTA (campo) performam melhor que imagens em fundo controlado (laboratório).

---

## Treinamento Autônomo (Etapa 2 — futuro)

Baseado no autoresearch do Karpathy, adaptado pra visão computacional:

### 3 Arquivos que Importam
```
training/
├── prepare.py    # Dataset loading, augmentation, evaluation (FIXO)
├── train.py      # EfficientNet + otimizador + loop (AGENTE MODIFICA)
└── program.md    # Instruções pro agente (HUMANO MODIFICA)
```

### Loop Autônomo
```
LOOP INFINITO:
1. Agente modifica train.py (muda LR, augmentation, arquitetura, batch)
2. git commit
3. Roda treinamento (5 min no Mac Mini via MPS)
4. Verifica acurácia no validation set
5. Se melhorou → git commit, mantém
6. Se piorou → git reset, descarta
7. Loga em results.tsv
8. Próximo experimento

~12 experimentos/hora × 8 horas = ~100 experimentos/noite
```

### Hiperparâmetros que o Agente Pode Mexer
- Learning rate (0.0001 — 0.01)
- Batch size (8, 16, 32)
- Arquitetura (EfficientNet-B0, B1, MobileNetV3)
- Augmentation (rotation, flip, color jitter, cutout, mixup)
- Optimizer (Adam, AdamW, SGD with momentum)
- Scheduler (cosine, step, one-cycle)
- Dropout rate (0.0 — 0.5)
- Image size (224, 256, 320)
- Número de epochs no budget de 5 min
- Freezing strategy (freeze backbone, fine-tune all, gradual unfreeze)

---

## Design da Plataforma Web

### Filosofia
- Dark mode padrão (consistente com Eli)
- Dashboard científico — limpo, profissional, data-driven
- Cards visuais com fotos das doenças
- Gráficos de progresso do dataset
- Sidebar com navegação clara

### Paleta de Cores
```
background: '#0B0F19'       # Azul escuro profundo
surface: '#131825'           # Cards
surfaceLight: '#1A2035'      # Hover
primary: '#10B981'           # Verde esmeralda (agricultura)
primaryLight: '#34D399'      # Verde claro
accent: '#F59E0B'            # Âmbar (alertas, severidade alta)
text: '#E2E8F0'              # Texto principal
textSecondary: '#94A3B8'     # Texto secundário
error: '#EF4444'             # Vermelho (doença crítica)
warning: '#F59E0B'           # Âmbar
success: '#10B981'           # Verde
```

---

## Entregáveis pro Tio

### Semana 1-2: Relatório
- Fichas técnicas por doença (visual, não só texto)
- Comparativo de fungicidas recomendados
- Mapa de incidência no Brasil (última safra)
- Links para papers mais relevantes

### Semana 3-4: Galeria + Dataset
- Galeria visual organizada por doença e estágio
- Interface de upload pra ele contribuir fotos
- Dashboard mostrando quantas imagens por classe

### Semana 5-8: Demo do Modelo
- Classificador funcional com 3-4 classes
- Demo: ele mostra foto → modelo identifica
- Métricas: acurácia, precision, recall por classe

### Perguntas pra Validar com Tio
1. Quais doenças mais diagnostica no campo?
2. Qual o custo de um diagnóstico errado?
3. Confiaria num modelo com X% de acurácia?
4. Prefere app ou WhatsApp (enviar foto e receber diagnóstico)?
5. Quanto pagaria por ferramenta dessas?
6. Conhece outros agrônomos que usariam?

---

## Modelo de Negócio (futuro)

| Produto | Público | Preço |
|--|--|--|
| App gratuito (limite diário) | Agrônomos individuais | R$ 0 |
| App PRO (ilimitado + histórico) | Profissionais | R$ 29,90/mês |
| API para integração | Cooperativas, revendas | Sob consulta |
| Relatórios regionais | Trading companies, seguradoras | R$ 5.000/mês |
| Consultoria (tio) | Propriedades > 1.000 ha | R$ 2.500/dia |

---

## Comandos Úteis

```bash
# Rodar dev server
cd ~/projetos/soja-research && npm run dev

# Abrir Claude Code no projeto
cd ~/projetos/soja-research && claude

# Buscar artigos (futuro script)
npm run scrape:articles "ferrugem soja"

# Baixar datasets (futuro script)
npm run scrape:images plantvillage

# Rodar treinamento (futuro, Etapa 2)
cd training && python train.py

# Usar DeepSeek pra resumir artigo
ollama run deepseek-r1:8b "Resuma este artigo sobre ferrugem asiática em soja: [abstract]"
```

---

## Convenções de Código

- TypeScript strict — sem `any`
- Componentes funcionais com hooks
- Nomes em inglês no código, strings em português na UI
- Temas via Tailwind config — nunca hardcode de cores
- APIs em `src/lib/api/` — cada fonte em arquivo separado
- Tipos em `src/types/` — interface pra cada entidade
- Dark mode padrão

---

## Contexto do Projeto

Projeto criado por **Gabriel Casa** (Grupo Casa, Balneário Camboriú/SC) em colaboração com tio agrônomo. Gabriel desenvolve usando Claude Code + modelos locais (DeepSeek-R1 8B, Qwen3 4B) no Mac Mini M4 16GB.

**Prioridade**: Eli (app saúde) é projeto #1. Soja Research é projeto #2, desenvolvido em paralelo quando possível.

**Inspiração**: autoresearch do Karpathy (pesquisa autônoma com agentes IA) adaptado para visão computacional agrícola.

---

*Última atualização: Março 2026*
*Versão: 1.0*
