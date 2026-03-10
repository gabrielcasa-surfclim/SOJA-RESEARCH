-- ============================================
-- SOJA RESEARCH — Schema Supabase (PostgreSQL)
-- ============================================

-- Extensão para busca textual
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================
-- 1. DOENÇAS-ALVO
-- ============================================
CREATE TABLE diseases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    scientific_name VARCHAR(200),
    symptoms TEXT[] DEFAULT '{}',
    severity VARCHAR(20) CHECK (severity IN ('baixa', 'média', 'alta')),
    conditions TEXT,
    management TEXT[] DEFAULT '{}',
    fungicides TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================
-- 2. IMAGENS CLASSIFICADAS POR DOENÇA
-- ============================================
CREATE TABLE disease_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    disease_id UUID NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    thumbnail_url TEXT,
    source VARCHAR(50) CHECK (source IN ('plantvillage', 'kaggle', 'field', 'upload', 'asdid', 'digipathos', 'doencasdeplantas', 'srin')),
    metadata JSONB DEFAULT '{}',
    validated BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_images_disease ON disease_images(disease_id);
CREATE INDEX idx_images_validated ON disease_images(validated);
CREATE INDEX idx_images_source ON disease_images(source);

-- ============================================
-- 3. ARTIGOS CIENTÍFICOS
-- ============================================
CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    authors TEXT[] DEFAULT '{}',
    abstract TEXT,
    doi VARCHAR(100) UNIQUE,
    source VARCHAR(50) CHECK (source IN ('pubmed', 'crossref', 'semanticscholar')),
    url TEXT,
    published_date DATE,
    methods TEXT[] DEFAULT '{}',
    accuracy FLOAT,
    dataset_used TEXT,
    summary TEXT,
    citation_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_articles_doi ON articles(doi);
CREATE INDEX idx_articles_source ON articles(source);
CREATE INDEX idx_articles_published ON articles(published_date DESC);
CREATE INDEX idx_articles_title ON articles USING GIN (title gin_trgm_ops);

-- ============================================
-- 4. RELAÇÃO ARTIGOS ↔ DOENÇAS (N:N)
-- ============================================
CREATE TABLE article_disease (
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    disease_id UUID NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
    relevance_score FLOAT DEFAULT 0.5 CHECK (relevance_score >= 0 AND relevance_score <= 1),
    PRIMARY KEY (article_id, disease_id)
);

CREATE INDEX idx_article_disease_article ON article_disease(article_id);
CREATE INDEX idx_article_disease_disease ON article_disease(disease_id);

-- ============================================
-- 5. FEEDBACK DO ESPECIALISTA
-- ============================================
CREATE TABLE image_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES disease_images(id) ON DELETE CASCADE,
    original_disease_id UUID REFERENCES diseases(id),
    suggested_disease_id UUID REFERENCES diseases(id),
    confidence VARCHAR(20) CHECK (confidence IN ('certeza', 'suspeita', 'não sei')),
    notes TEXT,
    gps_lat FLOAT,
    gps_lng FLOAT,
    weather VARCHAR(20) CHECK (weather IN ('sol', 'nublado', 'chuva')),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_feedback_image ON image_feedback(image_id);
CREATE INDEX idx_feedback_suggested ON image_feedback(suggested_disease_id);

-- ============================================
-- 6. EXPERIMENTOS DE TREINAMENTO
-- ============================================
CREATE TABLE training_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    config JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    commit_hash VARCHAR(7),
    val_accuracy FLOAT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_experiments_status ON training_experiments(status);
CREATE INDEX idx_experiments_accuracy ON training_experiments(val_accuracy DESC);
CREATE INDEX idx_experiments_created ON training_experiments(created_at DESC);

-- ============================================
-- VIEWS ÚTEIS
-- ============================================

-- Contagem de imagens por doença
CREATE OR REPLACE VIEW disease_image_counts AS
SELECT
    d.id,
    d.name,
    COUNT(di.id) AS image_count,
    COUNT(di.id) FILTER (WHERE di.validated = true) AS validated_count
FROM diseases d
LEFT JOIN disease_images di ON d.id = di.disease_id
GROUP BY d.id, d.name;

-- Contagem de artigos por doença
CREATE OR REPLACE VIEW disease_article_counts AS
SELECT
    d.id,
    d.name,
    COUNT(ad.article_id) AS article_count
FROM diseases d
LEFT JOIN article_disease ad ON d.id = ad.disease_id
GROUP BY d.id, d.name;

-- Melhor experimento por modelo
CREATE OR REPLACE VIEW best_experiments AS
SELECT DISTINCT ON (model_name)
    id,
    model_name,
    val_accuracy,
    config,
    metrics,
    completed_at
FROM training_experiments
WHERE status = 'completed' AND val_accuracy IS NOT NULL
ORDER BY model_name, val_accuracy DESC;

-- ============================================
-- SEED DATA: 7 DOENÇAS
-- ============================================
INSERT INTO diseases (name, scientific_name, symptoms, severity, conditions, management, fungicides) VALUES
(
    'Ferrugem Asiática',
    'Phakopsora pachyrhizi',
    ARRAY['Pústulas marrom-avermelhadas na parte inferior da folha', 'Lesões angulares delimitadas pelas nervuras', 'Desfolha prematura em casos severos', 'Urédias com esporulação visível'],
    'alta',
    'Temperatura entre 18-26°C com alta umidade relativa (>80%). Períodos prolongados de molhamento foliar (6+ horas). Disseminação por vento a longas distâncias.',
    ARRAY['Monitoramento constante a partir do fechamento das entrelinhas', 'Aplicação preventiva de fungicidas', 'Utilização de cultivares com resistência parcial', 'Respeitar o vazio sanitário', 'Rotação de princípios ativos'],
    ARRAY['Triazóis (epoxiconazol, protioconazol)', 'Estrobilurinas (azoxistrobina, piraclostrobina)', 'Carboxamidas (fluxapiroxade, benzovindiflupir)', 'Misturas multissítios (mancozebe, clorotalonil)']
),
(
    'Mancha-olho-de-rã',
    'Cercospora sojina',
    ARRAY['Lesões circulares com centro cinza-claro e borda marrom-avermelhada', 'Aspecto de "olho de rã"', 'Manchas foliares de 1-5mm de diâmetro', 'Pode afetar hastes e vagens em infecções severas'],
    'média',
    'Temperatura entre 25-30°C com alta umidade. Favorecida por chuvas frequentes e orvalho prolongado. Sobrevive em restos culturais e sementes infectadas.',
    ARRAY['Uso de sementes certificadas e tratadas', 'Rotação de culturas', 'Cultivares resistentes (gene Rcs3)', 'Eliminação de restos culturais', 'Fungicidas foliares preventivos'],
    ARRAY['Benzimidazóis (carbendazim)', 'Estrobilurinas (azoxistrobina)', 'Triazóis (flutriafol, ciproconazol)', 'Misturas triazol + estrobilurina']
),
(
    'Oídio',
    'Erysiphe diffusa',
    ARRAY['Pó branco/cinza na superfície superior das folhas', 'Pode cobrir caule e vagens', 'Folhas podem secar e cair prematuramente', 'Redução da área fotossintética'],
    'média',
    'Temperatura entre 18-24°C com umidade relativa moderada (50-70%). Favorecido por noites frescas e dias secos. Não necessita de molhamento foliar para infecção.',
    ARRAY['Cultivares resistentes', 'Espaçamento adequado entre plantas', 'Fungicidas ao primeiro sinal da doença', 'Evitar plantio adensado', 'Monitoramento frequente em períodos secos'],
    ARRAY['Triazóis (tebuconazol, difenoconazol)', 'Estrobilurinas (azoxistrobina, trifloxistrobina)', 'Enxofre (em formulações específicas)', 'Misturas triazol + estrobilurina']
),
(
    'Antracnose',
    'Colletotrichum truncatum',
    ARRAY['Manchas escuras irregulares em folhas', 'Necrose de pecíolos e hastes', 'Vagens com lesões deprimidas e escuras', 'Retenção foliar e haste verde em colheita'],
    'alta',
    'Temperatura entre 22-28°C com alta umidade e chuvas frequentes. Sobrevive em sementes e restos culturais. Disseminação por respingos de chuva.',
    ARRAY['Tratamento de sementes com fungicidas', 'Rotação de culturas (2-3 anos)', 'Uso de sementes sadias', 'Incorporação de restos culturais', 'Fungicidas foliares em condições favoráveis'],
    ARRAY['Carbendazim + tiram (tratamento de sementes)', 'Estrobilurinas (azoxistrobina)', 'Triazóis (difenoconazol)', 'Misturas com carboxamidas']
),
(
    'Mosaico',
    'BPMV / SMV',
    ARRAY['Padrão de mosaico verde claro/escuro nas folhas', 'Enrugamento e deformação foliar', 'Nanismo da planta', 'Manchas nas sementes (mancha café)', 'Redução no tamanho das vagens'],
    'média',
    'Transmitido por vetores (pulgões para SMV, coleópteros para BPMV). Também transmitido por sementes infectadas. Favorecido por alta população de vetores.',
    ARRAY['Uso de sementes livres de vírus', 'Controle de vetores (pulgões e coleópteros)', 'Cultivares resistentes/tolerantes', 'Eliminar plantas voluntárias', 'Monitoramento de populações de vetores'],
    ARRAY['Sem controle químico direto para o vírus', 'Inseticidas para controle de vetores (imidacloprido, tiametoxam)', 'Tratamento de sementes com inseticidas sistêmicos']
),
(
    'Mancha-alvo',
    'Corynespora cassiicola',
    ARRAY['Lesões com anéis concêntricos (formato de alvo)', 'Pontuação escura central com halos concêntricos', 'Manchas de 1-2cm que podem coalescer', 'Desfolha intensa em ataques severos'],
    'alta',
    'Temperatura entre 20-30°C com alta umidade relativa. Favorecida por chuvas frequentes e plantio direto (restos culturais). Fungo polífago que ataca várias culturas.',
    ARRAY['Rotação de culturas com não-hospedeiros', 'Cultivares menos suscetíveis', 'Fungicidas preventivos', 'Manejo de restos culturais', 'Evitar monocultura soja-soja'],
    ARRAY['Carboxamidas (boscalida, fluxapiroxade)', 'Estrobilurinas (piraclostrobina)', 'Triazóis (protioconazol)', 'Misturas multissítios como complemento']
),
(
    'Folha Saudável',
    NULL,
    ARRAY['Coloração verde uniforme', 'Sem lesões ou manchas', 'Textura normal da folha', 'Nervuras bem definidas'],
    NULL,
    'Condições ideais de cultivo com nutrição equilibrada, irrigação adequada e manejo fitossanitário preventivo.',
    ARRAY['Manter programa de adubação equilibrado', 'Monitoramento preventivo regular', 'Manejo integrado de pragas e doenças', 'Rotação de culturas'],
    ARRAY[]::TEXT[]
);
