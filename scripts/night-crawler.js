import { createClient } from '@supabase/supabase-js';
import {
  readFileSync,
  mkdirSync,
  existsSync,
  createWriteStream,
  readdirSync,
  unlinkSync,
} from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from 'stream/promises';
import { execSync } from 'child_process';

// ---------------------------------------------------------------------------
// Env & Supabase
// ---------------------------------------------------------------------------
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const envPath = resolve(ROOT, '.env');
const envVars = Object.fromEntries(
  readFileSync(envPath, 'utf-8')
    .split('\n')
    .filter((l) => l.includes('=') && !l.startsWith('#'))
    .map((l) => {
      const idx = l.indexOf('=');
      return [l.slice(0, idx).trim(), l.slice(idx + 1).trim()];
    })
);

const supabaseUrl = envVars.VITE_SUPABASE_URL;
const supabaseKey = envVars.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ VITE_SUPABASE_URL e VITE_SUPABASE_ANON_KEY são obrigatórios no .env');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// ---------------------------------------------------------------------------
// Doenças & termos de busca (variações pra rotacionar a cada rodada)
// ---------------------------------------------------------------------------
const diseases = [
  {
    id: 'ferrugem-asiatica',
    name: 'Ferrugem Asiática',
    searchTerms: [
      '"soybean rust" AND "deep learning"',
      '"Phakopsora pachyrhizi" AND "detection"',
      '"soybean rust" AND "image classification"',
      '"Asian soybean rust" AND "CNN"',
      '"Phakopsora pachyrhizi" AND "transfer learning"',
      '"soybean rust" AND "EfficientNet"',
    ],
  },
  {
    id: 'mancha-olho-de-ra',
    name: 'Mancha-olho-de-rã',
    searchTerms: [
      '"frogeye leaf spot" AND "soybean" AND "deep learning"',
      '"Cercospora sojina" AND "detection"',
      '"frogeye leaf spot" AND "image classification"',
      '"Cercospora sojina" AND "computer vision"',
      '"frogeye leaf spot" AND "CNN"',
    ],
  },
  {
    id: 'oidio',
    name: 'Oídio',
    searchTerms: [
      '"soybean powdery mildew" AND "image classification"',
      '"Erysiphe diffusa" AND "detection"',
      '"powdery mildew" AND "soybean" AND "deep learning"',
      '"powdery mildew" AND "soybean" AND "computer vision"',
      '"Erysiphe diffusa" AND "CNN"',
    ],
  },
  {
    id: 'antracnose',
    name: 'Antracnose',
    searchTerms: [
      '"soybean anthracnose" AND "deep learning"',
      '"Colletotrichum truncatum" AND "detection"',
      '"soybean anthracnose" AND "image classification"',
      '"Colletotrichum truncatum" AND "computer vision"',
      '"soybean anthracnose" AND "CNN"',
    ],
  },
  {
    id: 'mosaico',
    name: 'Mosaico',
    searchTerms: [
      '"soybean mosaic virus" AND "detection" AND "deep learning"',
      '"SMV" AND "soybean" AND "image classification"',
      '"soybean mosaic" AND "computer vision"',
      '"BPMV" AND "soybean" AND "detection"',
      '"soybean virus" AND "CNN"',
    ],
  },
  {
    id: 'mancha-alvo',
    name: 'Mancha-alvo',
    searchTerms: [
      '"target spot" AND "soybean" AND "deep learning"',
      '"Corynespora cassiicola" AND "detection"',
      '"target spot" AND "soybean" AND "computer vision"',
      '"Corynespora cassiicola" AND "image classification"',
      '"target spot" AND "soybean" AND "CNN"',
    ],
  },
  {
    id: 'folha-saudavel',
    name: 'Folha Saudável',
    searchTerms: [
      '"soybean disease detection" AND "transfer learning"',
      '"soybean leaf" AND "classification" AND "EfficientNet"',
      '"soybean disease" AND "CNN" AND "healthy"',
      '"soybean foliar disease" AND "deep learning"',
      '"soybean disease detection" AND "MobileNet"',
    ],
  },
];

// ---------------------------------------------------------------------------
// Busca por autores específicos (tio e colaboradores)
// ---------------------------------------------------------------------------
const AUTHOR_SEARCHES = [
  {
    name: 'Ricardo Trezzi Casa',
    pubmedQuery: '"Casa RT"[Author] OR "Trezzi Casa R"[Author] OR "Casa Ricardo"[Author]',
    crossrefQuery: 'Ricardo Trezzi Casa',
    semanticScholarQuery: 'Ricardo Trezzi Casa',
  },
  {
    name: 'Marta Casa Blum',
    pubmedQuery: '"Blum MC"[Author] OR "Casa Blum M"[Author] OR "Blum Marta"[Author]',
    crossrefQuery: 'Marta Casa Blum',
    semanticScholarQuery: 'Marta Casa Blum',
  },
];

// ---------------------------------------------------------------------------
// Digipathos / PDDB (EMBRAPA Dataverse) — download de imagens de soja
// https://www.redape.dados.embrapa.br/ — CC BY-NC 4.0, sem auth
// ---------------------------------------------------------------------------
const DIGIPATHOS_BASE = 'https://www.redape.dados.embrapa.br/api/access/datafile';
const DATA_DIR = join(ROOT, 'data', 'images');
const TMP_DIR = join(ROOT, 'data', 'tmp');

const DIGIPATHOS_CLASSES = [
  { fileId: 5953, zipName: 'ferrugem.zip', localName: 'digipathos_ferrugem', diseaseName: 'Ferrugem Asiática', label: 'Ferrugem' },
  { fileId: 6011, zipName: 'mancha_alvo.zip', localName: 'digipathos_mancha_alvo', diseaseName: 'Mancha-alvo', label: 'Mancha Alvo' },
  { fileId: 6097, zipName: 'oidio.zip', localName: 'digipathos_oidio', diseaseName: 'Oídio', label: 'Oídio' },
  { fileId: 6081, zipName: 'antracnose.zip', localName: 'digipathos_antracnose', diseaseName: 'Antracnose', label: 'Antracnose' },
  { fileId: 6060, zipName: 'mosaico.zip', localName: 'digipathos_mosaico', diseaseName: 'Mosaico', label: 'Folha Carijó (SMV)' },
  { fileId: 6088, zipName: 'saudavel.zip', localName: 'digipathos_saudavel', diseaseName: 'Folha Saudável', label: 'Saudável' },
  { fileId: 6033, zipName: 'cercospora.zip', localName: 'digipathos_cercospora', diseaseName: 'Mancha-olho-de-rã', label: 'Crestamento Cercospora' },
  // Cropped versions (recortadas, melhores pra treino)
  { fileId: 5993, zipName: 'ferrugem_crop.zip', localName: 'digipathos_ferrugem_crop', diseaseName: 'Ferrugem Asiática', label: 'Ferrugem (Cropped)' },
  { fileId: 5940, zipName: 'mancha_alvo_crop.zip', localName: 'digipathos_mancha_alvo_crop', diseaseName: 'Mancha-alvo', label: 'Mancha Alvo (Cropped)' },
  { fileId: 6051, zipName: 'oidio_crop.zip', localName: 'digipathos_oidio_crop', diseaseName: 'Oídio', label: 'Oídio (Cropped)' },
  { fileId: 6028, zipName: 'mosaico_crop.zip', localName: 'digipathos_mosaico_crop', diseaseName: 'Mosaico', label: 'Folha Carijó (Cropped)' },
  { fileId: 6000, zipName: 'cercospora_crop.zip', localName: 'digipathos_cercospora_crop', diseaseName: 'Mancha-olho-de-rã', label: 'Cercospora (Cropped)' },
];

// ---------------------------------------------------------------------------
// API — PubMed
// ---------------------------------------------------------------------------
const PUBMED_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils';

async function searchPubMed(query, maxResults = 20) {
  try {
    const searchUrl = `${PUBMED_BASE}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(query)}&retmax=${maxResults}&retmode=json`;
    const searchRes = await fetch(searchUrl);
    if (!searchRes.ok) return [];

    const searchData = await searchRes.json();
    const ids = searchData.esearchresult?.idlist ?? [];
    if (ids.length === 0) return [];

    const summaryUrl = `${PUBMED_BASE}/esummary.fcgi?db=pubmed&id=${ids.join(',')}&retmode=json`;
    const summaryRes = await fetch(summaryUrl);
    if (!summaryRes.ok) return [];

    const summaryData = await summaryRes.json();
    const results = summaryData.result;
    if (!results) return [];

    return ids
      .filter((id) => results[id])
      .map((id) => {
        const item = results[id];
        const authors = (item.authors ?? []).map((a) => a.name);
        const doi = (item.articleids ?? []).find((aid) => aid.idtype === 'doi')?.value;

        return {
          title: item.title ?? '',
          authors,
          abstract: '',
          doi: doi || null,
          source: 'pubmed',
          url: doi ? `https://doi.org/${doi}` : `https://pubmed.ncbi.nlm.nih.gov/${id}/`,
          published_date: item.pubdate ?? null,
        };
      });
  } catch (err) {
    console.error('  ⚠ PubMed erro:', err.message);
    return [];
  }
}

// ---------------------------------------------------------------------------
// API — CrossRef
// ---------------------------------------------------------------------------
async function searchCrossRef(query, maxResults = 20) {
  try {
    const url = `https://api.crossref.org/works?query=${encodeURIComponent(query)}&rows=${maxResults}`;
    const res = await fetch(url, {
      headers: { 'User-Agent': 'SojaResearch/1.0 (mailto:soja-research@example.com)' },
    });
    if (!res.ok) return [];

    const data = await res.json();
    const items = data.message?.items ?? [];

    return items.map((item) => {
      const authors = (item.author ?? []).map((a) =>
        [a.given, a.family].filter(Boolean).join(' ')
      );
      const dateParts = item.published?.['date-parts']?.[0];
      const publishedDate = dateParts
        ? dateParts.map((p) => String(p).padStart(2, '0')).join('-')
        : null;

      return {
        title: item.title?.[0] ?? '',
        authors,
        abstract: item.abstract ?? '',
        doi: item.DOI || null,
        source: 'crossref',
        url: item.DOI ? `https://doi.org/${item.DOI}` : null,
        published_date: publishedDate,
      };
    });
  } catch (err) {
    console.error('  ⚠ CrossRef erro:', err.message);
    return [];
  }
}

// ---------------------------------------------------------------------------
// API — Semantic Scholar
// ---------------------------------------------------------------------------
async function searchSemanticScholar(query, maxResults = 20) {
  try {
    const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=${maxResults}&fields=title,abstract,authors,year,externalIds,url`;
    const res = await fetch(url);
    if (!res.ok) return [];

    const data = await res.json();
    const papers = data.data ?? [];

    return papers.map((paper) => {
      const authors = (paper.authors ?? []).map((a) => a.name);
      const doi = paper.externalIds?.DOI || null;

      return {
        title: paper.title ?? '',
        authors,
        abstract: paper.abstract ?? '',
        doi,
        source: 'semanticscholar',
        url: doi ? `https://doi.org/${doi}` : paper.url || null,
        published_date: paper.year ? `${paper.year}-01-01` : null,
      };
    });
  } catch (err) {
    console.error('  ⚠ Semantic Scholar erro:', err.message);
    return [];
  }
}

// ---------------------------------------------------------------------------
// CrossRef — busca por autor
// ---------------------------------------------------------------------------
async function searchCrossRefByAuthor(authorName, maxResults = 50) {
  try {
    const url = `https://api.crossref.org/works?query.author=${encodeURIComponent(authorName)}&rows=${maxResults}&sort=published&order=desc`;
    const res = await fetch(url, {
      headers: { 'User-Agent': 'SojaResearch/1.0 (mailto:soja-research@example.com)' },
    });
    if (!res.ok) return [];

    const data = await res.json();
    const items = data.message?.items ?? [];

    return items.map((item) => {
      const authors = (item.author ?? []).map((a) =>
        [a.given, a.family].filter(Boolean).join(' ')
      );
      const dateParts = item.published?.['date-parts']?.[0];
      const publishedDate = dateParts
        ? dateParts.map((p) => String(p).padStart(2, '0')).join('-')
        : null;

      return {
        title: item.title?.[0] ?? '',
        authors,
        abstract: item.abstract ?? '',
        doi: item.DOI || null,
        source: 'crossref',
        url: item.DOI ? `https://doi.org/${item.DOI}` : null,
        published_date: publishedDate,
      };
    });
  } catch (err) {
    console.error('  ⚠ CrossRef author search erro:', err.message);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Supabase — verificar DOI e salvar
// ---------------------------------------------------------------------------
async function getDiseaseUUID(diseaseSlug) {
  const { data } = await supabase
    .from('diseases')
    .select('id')
    .eq('id', diseaseSlug)
    .single();
  return data?.id ?? null;
}

// Cache de disease UUIDs
const diseaseUUIDCache = new Map();

async function resolveDiseaseId(slug) {
  if (diseaseUUIDCache.has(slug)) return diseaseUUIDCache.get(slug);
  const uuid = await getDiseaseUUID(slug);
  if (uuid) diseaseUUIDCache.set(slug, uuid);
  return uuid;
}

async function doiExists(doi) {
  if (!doi) return false;
  const { data } = await supabase
    .from('articles')
    .select('id')
    .eq('doi', doi)
    .limit(1);
  return data && data.length > 0;
}

async function titleExists(title) {
  if (!title) return false;
  const { data } = await supabase
    .from('articles')
    .select('id')
    .eq('title', title)
    .limit(1);
  return data && data.length > 0;
}

async function saveArticle(article, diseaseId) {
  const { data, error } = await supabase
    .from('articles')
    .insert({
      title: article.title,
      authors: article.authors,
      abstract: article.abstract || null,
      doi: article.doi,
      source: article.source,
      url: article.url,
      published_date: normalizeDate(article.published_date),
    })
    .select('id')
    .single();

  if (error) {
    if (error.code === '23505') return null;
    console.error('  ⚠ Erro ao salvar artigo:', error.message);
    return null;
  }

  if (data?.id && diseaseId) {
    const { error: relError } = await supabase
      .from('article_disease')
      .insert({
        article_id: data.id,
        disease_id: diseaseId,
        relevance_score: 0.8,
      });

    if (relError && relError.code !== '23505') {
      console.error('  ⚠ Erro ao criar relação:', relError.message);
    }
  }

  return data?.id ?? null;
}

/** Save article without disease association (for author searches) */
async function saveArticleNoDiseaseLink(article) {
  const { data, error } = await supabase
    .from('articles')
    .insert({
      title: article.title,
      authors: article.authors,
      abstract: article.abstract || null,
      doi: article.doi,
      source: article.source,
      url: article.url,
      published_date: normalizeDate(article.published_date),
    })
    .select('id')
    .single();

  if (error) {
    if (error.code === '23505') return null;
    console.error('  ⚠ Erro ao salvar artigo:', error.message);
    return null;
  }

  return data?.id ?? null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const MONTH_MAP = {
  jan: '01', january: '01',
  feb: '02', february: '02',
  mar: '03', march: '03',
  apr: '04', april: '04',
  may: '05',
  jun: '06', june: '06',
  jul: '07', july: '07',
  aug: '08', august: '08',
  sep: '09', september: '09',
  oct: '10', october: '10',
  nov: '11', november: '11',
  dec: '12', december: '12',
};

/**
 * Normaliza datas em formatos variados para "YYYY-MM-DD" válido pro PostgreSQL.
 * Exemplos: "2024" → "2024-01-01", "2025 Dec" → "2025-12-01", "2026-06" → "2026-06-01"
 * Retorna null se não conseguir parsear.
 */
function normalizeDate(raw) {
  if (!raw) return null;
  const s = String(raw).trim();
  if (!s) return null;

  // Already valid YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;

  // YYYY-MM (e.g. "2026-06")
  if (/^\d{4}-\d{1,2}$/.test(s)) {
    const [y, m] = s.split('-');
    return `${y}-${m.padStart(2, '0')}-01`;
  }

  // Just a year: "2024", "1991"
  if (/^\d{4}$/.test(s)) return `${s}-01-01`;

  // "YYYY Mon" or "Mon YYYY" (e.g. "2025 Dec", "Oct 2024")
  const monthYearMatch = s.match(/^(\d{4})\s+([A-Za-z]+)$/) || s.match(/^([A-Za-z]+)\s+(\d{4})$/);
  if (monthYearMatch) {
    const parts = monthYearMatch;
    const yearStr = parts[1].match(/^\d{4}$/) ? parts[1] : parts[2];
    const monStr = parts[1].match(/^\d{4}$/) ? parts[2] : parts[1];
    const month = MONTH_MAP[monStr.toLowerCase()];
    if (month && yearStr) return `${yearStr}-${month}-01`;
  }

  // PubMed format: "2024 Oct 15" or "2024 Oct"
  const pubmedMatch = s.match(/^(\d{4})\s+([A-Za-z]+)\s+(\d{1,2})$/);
  if (pubmedMatch) {
    const month = MONTH_MAP[pubmedMatch[2].toLowerCase()];
    if (month) return `${pubmedMatch[1]}-${month}-${pubmedMatch[3].padStart(2, '0')}`;
  }

  // Full date with slashes or dots: "15/10/2024", "2024.10.15"
  const slashMatch = s.match(/^(\d{4})[./](\d{1,2})[./](\d{1,2})$/);
  if (slashMatch) return `${slashMatch[1]}-${slashMatch[2].padStart(2, '0')}-${slashMatch[3].padStart(2, '0')}`;

  // Last resort: try Date.parse
  const parsed = Date.parse(s);
  if (!isNaN(parsed)) {
    const d = new Date(parsed);
    const y = d.getUTCFullYear();
    if (y >= 1900 && y <= 2100) {
      return `${y}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
    }
  }

  return null;
}

/**
 * Verifica se pelo menos um autor contém "Casa" ou "Blum" no nome.
 */
function hasTargetAuthor(authors) {
  if (!authors || authors.length === 0) return false;
  return authors.some((name) => /\bcasa\b/i.test(name) || /\bblum\b/i.test(name));
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function timestamp() {
  return new Date().toLocaleString('pt-BR', { timeZone: 'America/Sao_Paulo' });
}

// ---------------------------------------------------------------------------
// Busca por autores (roda 1x por rodada)
// ---------------------------------------------------------------------------
async function crawlAuthors() {
  console.log('\n🧑‍🔬 Buscando artigos por autor...');
  let saved = 0;

  for (const author of AUTHOR_SEARCHES) {
    console.log(`\n   👤 ${author.name}`);

    // --- PubMed (author query) ---
    console.log('   📚 PubMed...');
    const pubmedArticles = await searchPubMed(author.pubmedQuery, 50);
    console.log(`   → ${pubmedArticles.length} resultados`);

    for (const article of pubmedArticles) {
      if (await doiExists(article.doi)) continue;
      if (!article.doi && await titleExists(article.title)) continue;
      const id = await saveArticleNoDiseaseLink(article);
      if (id) {
        saved++;
        console.log(`   ✅ ${article.title.slice(0, 70)}...`);
      }
    }

    await sleep(2000);

    // --- CrossRef (author search) ---
    console.log('   📚 CrossRef...');
    const crossrefArticlesRaw = await searchCrossRefByAuthor(author.name, 50);
    const crossrefArticles = crossrefArticlesRaw.filter((a) => hasTargetAuthor(a.authors));
    console.log(`   → ${crossrefArticlesRaw.length} resultados, ${crossrefArticles.length} após filtro de autor`);

    for (const article of crossrefArticles) {
      if (await doiExists(article.doi)) continue;
      if (!article.doi && await titleExists(article.title)) continue;
      const id = await saveArticleNoDiseaseLink(article);
      if (id) {
        saved++;
        console.log(`   ✅ ${article.title.slice(0, 70)}...`);
      }
    }

    await sleep(2000);

    // --- Semantic Scholar ---
    console.log('   📚 Semantic Scholar...');
    const ssArticlesRaw = await searchSemanticScholar(author.semanticScholarQuery, 50);
    const ssArticles = ssArticlesRaw.filter((a) => hasTargetAuthor(a.authors));
    console.log(`   → ${ssArticlesRaw.length} resultados, ${ssArticles.length} após filtro de autor`);

    for (const article of ssArticles) {
      if (await doiExists(article.doi)) continue;
      if (!article.doi && await titleExists(article.title)) continue;
      const id = await saveArticleNoDiseaseLink(article);
      if (id) {
        saved++;
        console.log(`   ✅ ${article.title.slice(0, 70)}...`);
      }
    }

    await sleep(2000);
  }

  console.log(`\n   🧑‍🔬 Autores: ${saved} artigos novos salvos.`);
  return saved;
}

// ---------------------------------------------------------------------------
// Digipathos image download (roda 1x, no início)
// ---------------------------------------------------------------------------
async function downloadDigipathos() {
  console.log('\n🌿 Digipathos / PDDB (EMBRAPA) — Download de imagens de soja');
  console.log('   Fonte: https://www.redape.dados.embrapa.br/');
  console.log('   Licença: CC BY-NC 4.0\n');

  // Lookup disease IDs
  const { data: dbDiseases } = await supabase.from('diseases').select('id, name');
  if (!dbDiseases) {
    console.error('   ❌ Erro ao buscar doenças no Supabase.');
    return;
  }
  const diseaseMap = new Map(dbDiseases.map((d) => [d.name, d.id]));

  mkdirSync(TMP_DIR, { recursive: true });

  let totalImages = 0;
  let totalRegistered = 0;

  for (const cls of DIGIPATHOS_CLASSES) {
    const diseaseId = diseaseMap.get(cls.diseaseName);
    if (!diseaseId) {
      console.log(`   ⚠ "${cls.diseaseName}" não encontrada no Supabase. Pulando ${cls.label}.`);
      continue;
    }

    const localDir = join(DATA_DIR, cls.localName);

    // Check if already downloaded
    if (existsSync(localDir)) {
      const existing = readdirSync(localDir).filter((f) =>
        /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i.test(f)
      );
      if (existing.length > 0) {
        console.log(`   ✓ ${cls.label}: ${existing.length} imagens já existem.`);

        // Register any unregistered
        const reg = await registerDigipathosImages(localDir, cls.localName, diseaseId);
        totalRegistered += reg;
        if (reg > 0) console.log(`     → ${reg} novas registradas no Supabase.`);
        continue;
      }
    }

    mkdirSync(localDir, { recursive: true });

    const zipPath = join(TMP_DIR, cls.zipName);
    const downloadUrl = `${DIGIPATHOS_BASE}/${cls.fileId}`;

    console.log(`   📥 ${cls.label} (fileId: ${cls.fileId})...`);

    // Download
    if (!existsSync(zipPath)) {
      try {
        execSync(`curl -L -C - --progress-bar -o "${zipPath}" "${downloadUrl}"`, {
          stdio: 'inherit',
        });
      } catch (err) {
        console.error(`   ❌ Erro ao baixar ${cls.label}: ${err.message}`);
        continue;
      }
    }

    // Extract — use -j to flatten, handling nested folders
    console.log(`   📦 Extraindo...`);
    try {
      execSync(`unzip -q -o "${zipPath}" -d "${localDir}"`, { stdio: 'inherit' });

      // Move images from subdirectories to root level
      try {
        execSync(
          `find "${localDir}" -mindepth 2 -type f \\( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \\) -exec mv -n {} "${localDir}/" \\;`,
          { stdio: 'inherit' }
        );
        // Remove empty subdirectories
        execSync(`find "${localDir}" -mindepth 1 -type d -empty -delete 2>/dev/null || true`, {
          stdio: 'inherit',
        });
      } catch {
        // fine if no nested dirs
      }

      // Remove non-image files
      for (const f of readdirSync(localDir)) {
        const full = join(localDir, f);
        if (!/\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i.test(f)) {
          try { unlinkSync(full); } catch { /* dir or special */ }
        }
      }

      const extracted = readdirSync(localDir).filter((f) =>
        /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i.test(f)
      );
      totalImages += extracted.length;
      console.log(`   ✓ ${extracted.length} imagens extraídas.`);
    } catch (err) {
      console.error(`   ❌ Erro ao extrair: ${err.message}`);
      continue;
    }

    // Register in Supabase
    const reg = await registerDigipathosImages(localDir, cls.localName, diseaseId);
    totalRegistered += reg;
    console.log(`   → ${reg} registradas no Supabase.`);

    // Delete zip
    try { unlinkSync(zipPath); } catch { /* ok */ }
  }

  console.log(`\n🌿 Digipathos concluído: ${totalImages} imagens, ${totalRegistered} registros novos.\n`);
}

async function registerDigipathosImages(localDir, localName, diseaseId) {
  const files = readdirSync(localDir).filter((f) =>
    /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i.test(f)
  );
  if (files.length === 0) return 0;

  const { data: existing } = await supabase
    .from('disease_images')
    .select('url')
    .eq('disease_id', diseaseId)
    .eq('source', 'digipathos');

  const existingUrls = new Set((existing || []).map((e) => e.url));

  const newRecords = files
    .map((f) => {
      const localPath = `data/images/${localName}/${f}`;
      if (existingUrls.has(localPath)) return null;
      return {
        disease_id: diseaseId,
        url: localPath,
        source: 'digipathos',
        validated: true, // labeled by EMBRAPA pathologists
        metadata: {
          original_name: f,
          dataset: 'PDDB/Digipathos',
          doi: '10.48432/XA1OVL',
        },
      };
    })
    .filter(Boolean);

  if (newRecords.length === 0) return 0;

  let registered = 0;
  const BATCH = 100;
  for (let i = 0; i < newRecords.length; i += BATCH) {
    const batch = newRecords.slice(i, i + BATCH);
    const { error } = await supabase.from('disease_images').insert(batch);
    if (error) {
      console.error(`   ⚠ Erro ao inserir batch: ${error.message}`);
    } else {
      registered += batch.length;
    }
  }

  return registered;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
async function run() {
  console.log('🌱 Night Crawler — Soja Research');
  console.log(`   Início: ${timestamp()}`);
  console.log(`   Supabase: ${supabaseUrl}`);
  console.log('   Ctrl+C para parar\n');

  // --- Download Digipathos images (1x, no início) ---
  await downloadDigipathos();

  let totalFound = 0;
  let totalSaved = 0;
  let round = 0;

  while (true) {
    round++;
    console.log(`\n${'='.repeat(60)}`);
    console.log(`📡 RODADA ${round} — ${timestamp()}`);
    console.log(`${'='.repeat(60)}`);

    let roundFound = 0;
    let roundSaved = 0;

    // --- Busca por autores (1x por rodada) ---
    const authorSaved = await crawlAuthors();
    roundSaved += authorSaved;

    await sleep(5000);

    // --- Busca por doença ---
    for (const disease of diseases) {
      const termIndex = (round - 1) % disease.searchTerms.length;
      const query = disease.searchTerms[termIndex];

      console.log(`\n🔍 ${disease.name}`);
      console.log(`   Termo: ${query}`);

      const diseaseId = await resolveDiseaseId(disease.id);

      // --- PubMed ---
      console.log('   📚 Buscando PubMed...');
      const pubmedArticles = await searchPubMed(query);
      console.log(`   → ${pubmedArticles.length} resultados`);

      for (const article of pubmedArticles) {
        if (await doiExists(article.doi)) continue;
        if (!article.doi && await titleExists(article.title)) continue;

        const savedId = await saveArticle(article, diseaseId);
        if (savedId) {
          roundSaved++;
          console.log(`   ✅ Novo: ${article.title.slice(0, 70)}...`);
        }
      }
      roundFound += pubmedArticles.length;

      await sleep(2000);

      // --- CrossRef ---
      console.log('   📚 Buscando CrossRef...');
      const crossrefArticles = await searchCrossRef(query);
      console.log(`   → ${crossrefArticles.length} resultados`);

      for (const article of crossrefArticles) {
        if (await doiExists(article.doi)) continue;
        if (!article.doi && await titleExists(article.title)) continue;

        const savedId = await saveArticle(article, diseaseId);
        if (savedId) {
          roundSaved++;
          console.log(`   ✅ Novo: ${article.title.slice(0, 70)}...`);
        }
      }
      roundFound += crossrefArticles.length;

      // Esperar 30s entre cada doença pra respeitar rate limits
      console.log('   ⏳ Aguardando 30s...');
      await sleep(30_000);
    }

    totalFound += roundFound;
    totalSaved += roundSaved;

    console.log(`\n${'─'.repeat(60)}`);
    console.log(`📊 Rodada ${round} finalizada — ${timestamp()}`);
    console.log(`   Encontrados nesta rodada: ${roundFound}`);
    console.log(`   Salvos nesta rodada:      ${roundSaved}`);
    console.log(`   Total encontrados:        ${totalFound}`);
    console.log(`   Total salvos:             ${totalSaved}`);
    console.log(`${'─'.repeat(60)}`);

    // Pausa de 5 min entre rodadas completas
    console.log('\n💤 Próxima rodada em 5 minutos...');
    await sleep(5 * 60 * 1000);
  }
}

run().catch((err) => {
  console.error('💀 Erro fatal:', err);
  process.exit(1);
});
