/**
 * download-doencasdeplantas.js — Baixa fotos de soja do site doencasdeplantas.com.br
 *
 * Descobre TODAS as doenças dinamicamente a partir da página do site (31+).
 * As que mapeiam pras nossas 7 classes são registradas no Supabase.
 * As demais ficam como pastas extras (doencasdeplantas_{slug}) para uso futuro.
 *
 * O site expõe uma API interna POST /getDiseasesImages que retorna a lista de imagens
 * por doença. Precisamos do CSRF token (cookies XSRF-TOKEN + session) para usar.
 *
 * Uso:
 *   node scripts/download-doencasdeplantas.js [--all | --doencas slug1,slug2,...]
 *
 * Exemplos:
 *   node scripts/download-doencasdeplantas.js --all
 *   node scripts/download-doencasdeplantas.js --doencas ferrugem-asiatica,oidio,mancha-alvo
 */

import { createClient } from '@supabase/supabase-js';
import {
  readFileSync,
  mkdirSync,
  existsSync,
  createWriteStream,
  readdirSync,
} from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from 'stream/promises';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const DATA_DIR = join(ROOT, 'data', 'images');

const BASE_URL = 'https://www.doencasdeplantas.com.br';
const PLANT_ID = '42'; // soja
const CONCURRENCY = 3; // gentil com o servidor

// ---------------------------------------------------------------------------
// Mapeamento: slug do site -> classe no prepare.py / nome no Supabase
// Somente doenças que mapeiam pras nossas classes conhecidas.
// Doenças que não estão aqui serão baixadas mas não registradas no Supabase.
// ---------------------------------------------------------------------------
const SLUG_TO_CLASS = {
  'ferrugem-asiatica': 'Ferrugem Asiática',
  'oidio': 'Oídio',
  'mancha-alvo': 'Mancha-alvo',
  'mancha-olho-de-ra': 'Mancha-olho-de-rã',
  'antracnose': 'Antracnose',
  'virose-do-mosaico-comum': 'Mosaico',
  'virose-do-mosaico-rugoso': 'Mosaico',
  'cercosporiose': 'Cercospora',
};

// ---------------------------------------------------------------------------
// Env & Supabase
// ---------------------------------------------------------------------------
const envPath = resolve(ROOT, '.env');
let supabase = null;

if (existsSync(envPath)) {
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

  if (supabaseUrl && supabaseKey) {
    supabase = createClient(supabaseUrl, supabaseKey);
  }
}

if (!supabase) {
  console.warn('Supabase nao configurado (.env). Imagens serao baixadas mas nao registradas.\n');
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function asyncPool(concurrency, items, fn) {
  const executing = new Set();
  const results = [];
  for (const item of items) {
    const p = fn(item).then((r) => {
      executing.delete(p);
      return r;
    });
    executing.add(p);
    results.push(p);
    if (executing.size >= concurrency) await Promise.race(executing);
  }
  return Promise.all(results);
}

function slugToFolder(slug) {
  return `doencasdeplantas_${slug.replace(/-/g, '_')}`;
}

// ---------------------------------------------------------------------------
// Discover all diseases from the site HTML
// ---------------------------------------------------------------------------
async function discoverDiseases(html) {
  // The site embeds disease data as JS objects in the page.
  // We parse data-disease="ID" attributes and disease names/slugs.
  // Strategy: find the allDiseases JSON or parse cards from HTML.

  const diseases = [];
  const seen = new Set();

  // Pattern 1: Parse from embedded JS array (allDiseases or similar)
  // Look for objects with id, name_pt, url_pt
  const jsonRegex = /"id"\s*:\s*(\d+)\s*,\s*"name_pt"\s*:\s*"([^"]+)"\s*,\s*"url_pt"\s*:\s*"([^"]+)"/g;
  let match;
  while ((match = jsonRegex.exec(html)) !== null) {
    const [, id, namePt, urlPt] = match;
    if (seen.has(id)) continue;
    seen.add(id);
    diseases.push({ siteId: id, name: namePt, slug: urlPt });
  }

  // Pattern 2: Fallback — parse data-disease cards
  if (diseases.length === 0) {
    const cardRegex = /data-disease="(\d+)"[\s\S]*?<h\d[^>]*>\s*(.*?)\s*<\/h\d>/gi;
    while ((match = cardRegex.exec(html)) !== null) {
      const [, id, name] = match;
      if (seen.has(id)) continue;
      seen.add(id);
      const slug = name.toLowerCase()
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      diseases.push({ siteId: id, name, slug });
    }
  }

  // Filter out non-disease entries (like "Detalhes" id=2)
  return diseases.filter((d) => d.slug !== 'detalhes' && d.name.toLowerCase() !== 'detalhes');
}

// ---------------------------------------------------------------------------
// CSRF / Session + disease discovery
// ---------------------------------------------------------------------------
async function getSession() {
  console.log('Obtendo sessao, CSRF token e lista de doencas...');

  const res = await fetch(`${BASE_URL}/doencas-de-plantas/soja-42`, {
    headers: { 'User-Agent': 'soja-research-downloader/1.0' },
  });

  if (!res.ok) throw new Error(`Falha ao acessar site: ${res.status}`);

  const cookies = res.headers.getSetCookie?.() || [];
  const cookieMap = {};
  for (const c of cookies) {
    const [pair] = c.split(';');
    const [name, ...rest] = pair.split('=');
    cookieMap[name.trim()] = rest.join('=').trim();
  }

  const xsrfToken = cookieMap['XSRF-TOKEN'];
  const sessionCookie = cookieMap['doenca_de_plantas_session'];

  if (!xsrfToken) throw new Error('XSRF-TOKEN nao encontrado nos cookies');

  const html = await res.text();
  const siteDiseases = await discoverDiseases(html);

  return {
    xsrfToken: decodeURIComponent(xsrfToken),
    cookieHeader: `XSRF-TOKEN=${xsrfToken}; doenca_de_plantas_session=${sessionCookie}`,
    siteDiseases,
  };
}

// ---------------------------------------------------------------------------
// Fetch images from API
// ---------------------------------------------------------------------------
async function fetchDiseaseImages(session, siteId, slug) {
  const res = await fetch(`${BASE_URL}/getDiseasesImages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-XSRF-TOKEN': session.xsrfToken,
      'Cookie': session.cookieHeader,
      'User-Agent': 'soja-research-downloader/1.0',
      'X-Requested-With': 'XMLHttpRequest',
      'Referer': `${BASE_URL}/doencas-de-plantas/soja-42`,
    },
    body: JSON.stringify({ plant_id: PLANT_ID, disease_id: String(siteId) }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API retornou ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  if (data.status !== 1 || !Array.isArray(data.images)) {
    throw new Error(`Resposta inesperada: ${JSON.stringify(data).slice(0, 200)}`);
  }

  // The API sometimes returns images from wrong paths (e.g. cercosporiose returning mofo_branco files).
  // Filter: keep images whose filename loosely matches the disease slug.
  // We extract keywords from the slug and check if the filename contains at least one.
  const slugWords = slug.replace(/-/g, '_').split('_').filter((w) => w.length >= 4);
  const validImages = [];
  const skippedImages = [];

  for (const img of data.images) {
    if (!img.image) continue;
    const fname = img.image.toLowerCase().replace(/-/g, '_');
    // Accept if filename contains any significant slug keyword (handles typos like "ferruem" for "ferrugem")
    const matches = slugWords.some((word) => fname.includes(word.slice(0, 4)));
    if (matches) {
      validImages.push(img);
    } else {
      skippedImages.push(img.image);
    }
  }

  if (skippedImages.length > 0) {
    console.log(`  Filtradas ${skippedImages.length} imagens com nomes inconsistentes (ex: ${skippedImages[0]})`);
  }

  return validImages;
}

// ---------------------------------------------------------------------------
// Download image file
// ---------------------------------------------------------------------------
async function downloadFile(url, destPath) {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'soja-research-downloader/1.0' },
  });
  if (!res.ok) throw new Error(`Download failed ${res.status}: ${url}`);
  const ws = createWriteStream(destPath);
  await pipeline(res.body, ws);
}

// ---------------------------------------------------------------------------
// Register images in Supabase
// ---------------------------------------------------------------------------
async function registerImages(localDir, localFolder, diseaseId) {
  if (!supabase || !diseaseId) return 0;

  const files = readdirSync(localDir).filter((f) =>
    /\.(jpg|jpeg|png|webp)$/i.test(f)
  );
  if (files.length === 0) return 0;

  // Check already registered
  const { data: existing } = await supabase
    .from('disease_images')
    .select('url')
    .eq('disease_id', diseaseId)
    .eq('source', 'doencasdeplantas');

  const existingUrls = new Set((existing || []).map((e) => e.url));

  const newRecords = files
    .map((f) => {
      const localPath = `data/images/${localFolder}/${f}`;
      if (existingUrls.has(localPath)) return null;
      return {
        disease_id: diseaseId,
        url: localPath,
        source: 'doencasdeplantas',
        validated: false,
        metadata: { original_name: f, dataset: 'doencasdeplantas.com.br' },
      };
    })
    .filter(Boolean);

  if (newRecords.length === 0) return 0;

  const BATCH = 50;
  let registered = 0;
  for (let i = 0; i < newRecords.length; i += BATCH) {
    const batch = newRecords.slice(i, i + BATCH);
    const { error } = await supabase.from('disease_images').insert(batch);
    if (error) {
      console.error(`    Erro Supabase: ${error.message}`);
    } else {
      registered += batch.length;
    }
  }
  return registered;
}

// ---------------------------------------------------------------------------
// Download one disease
// ---------------------------------------------------------------------------
async function downloadDisease(disease, session, supabaseDiseaseMap) {
  const { slug, siteId, name } = disease;
  const localFolder = slugToFolder(slug);
  const localDir = join(DATA_DIR, localFolder);
  const diseaseName = SLUG_TO_CLASS[slug] || null;
  const mapped = diseaseName ? `-> ${diseaseName}` : '(extra)';

  mkdirSync(localDir, { recursive: true });

  console.log(`\n--- ${name} [${slug}] ${mapped} ---`);

  // Fetch image list from API
  let images;
  try {
    images = await fetchDiseaseImages(session, siteId, slug);
  } catch (err) {
    console.error(`  Erro ao listar imagens: ${err.message}`);
    return { downloaded: 0, registered: 0, total: 0, errors: 0 };
  }

  if (images.length === 0) {
    console.log('  0 imagens encontradas.');
    return { downloaded: 0, registered: 0, total: 0, errors: 0 };
  }

  console.log(`  ${images.length} imagens encontradas.`);

  // Build download list
  const toDownload = [];
  for (const img of images) {
    const filename = img.image;
    if (!filename) continue;

    const destPath = join(localDir, filename);
    if (existsSync(destPath)) continue;

    const url = `${BASE_URL}/img/uploads/plants_or_seeds/soja-42/${slug}/${encodeURIComponent(filename)}`;
    toDownload.push({ url, destPath, filename });
  }

  if (toDownload.length === 0) {
    console.log('  Todas ja baixadas.');
  } else {
    console.log(`  ${toDownload.length} novas para baixar.`);
  }

  // Download with concurrency
  let downloaded = 0;
  let errors = 0;
  await asyncPool(CONCURRENCY, toDownload, async (item) => {
    try {
      await downloadFile(item.url, item.destPath);
      downloaded++;
      if (downloaded % 10 === 0 || downloaded === toDownload.length) {
        console.log(`  Baixadas: ${downloaded}/${toDownload.length}`);
      }
    } catch (err) {
      errors++;
      if (errors <= 3) {
        console.error(`  Erro: ${item.filename}: ${err.message}`);
      } else if (errors === 4) {
        console.error(`  ... (suprimindo erros adicionais)`);
      }
    }
    await sleep(200);
  });

  if (errors > 0) console.log(`  ${errors} erros de download.`);

  // Register in Supabase (only mapped diseases)
  const diseaseId = diseaseName ? supabaseDiseaseMap.get(diseaseName) : null;
  let registered = 0;
  if (diseaseId) {
    console.log('  Registrando no Supabase...');
    registered = await registerImages(localDir, localFolder, diseaseId);
    console.log(`  ${registered} novas registradas.`);
  }

  const totalFiles = readdirSync(localDir).filter((f) => /\.(jpg|jpeg|png|webp)$/i.test(f)).length;
  console.log(`  Total na pasta: ${totalFiles} imagens.`);

  return { downloaded, registered, total: images.length, errors };
}

// ---------------------------------------------------------------------------
// CLI & Main
// ---------------------------------------------------------------------------
async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Uso: node scripts/download-doencasdeplantas.js [opcoes]

Opcoes:
  --all                              Baixar TODAS as doencas do site
  --doencas slug1,slug2,...          Baixar apenas doencas especificas

Os slugs sao descobertos automaticamente do site.
Doencas mapeadas pras 7 classes sao registradas no Supabase.
Demais ficam como pastas extras em data/images/doencasdeplantas_{slug}/.

Exemplos:
  node scripts/download-doencasdeplantas.js --all
  node scripts/download-doencasdeplantas.js --doencas ferrugem-asiatica,oidio,mancha-alvo
`);
    process.exit(0);
  }

  const doAll = args.includes('--all');

  let selectedSlugs = null;
  const doencasIdx = args.indexOf('--doencas');
  if (doencasIdx !== -1 && args[doencasIdx + 1]) {
    selectedSlugs = new Set(args[doencasIdx + 1].split(',').map((s) => s.trim()));
  }

  if (!doAll && !selectedSlugs) {
    console.log('Use --all ou --doencas. Veja --help para opcoes.');
    process.exit(0);
  }

  console.log('========================================');
  console.log('  DOENCAS DE PLANTAS — Downloader');
  console.log('  doencasdeplantas.com.br/soja');
  console.log('========================================\n');

  // Get CSRF session + discover diseases
  let session;
  try {
    session = await getSession();
  } catch (err) {
    console.error(`Erro ao obter sessao: ${err.message}`);
    process.exit(1);
  }

  const allDiseases = session.siteDiseases;
  console.log(`${allDiseases.length} doencas encontradas no site.\n`);

  // Filter if needed
  const diseases = doAll
    ? allDiseases
    : allDiseases.filter((d) => selectedSlugs.has(d.slug));

  if (diseases.length === 0) {
    console.error('Nenhuma doenca encontrada. Slugs disponiveis:');
    allDiseases.forEach((d) => console.log(`  ${d.slug} (${d.name})`));
    process.exit(1);
  }

  const mapped = diseases.filter((d) => SLUG_TO_CLASS[d.slug]);
  const extra = diseases.filter((d) => !SLUG_TO_CLASS[d.slug]);
  console.log(`Mapeadas: ${mapped.length} | Extras: ${extra.length} | Total: ${diseases.length}\n`);

  // List all
  for (const d of diseases) {
    const tag = SLUG_TO_CLASS[d.slug] ? `-> ${SLUG_TO_CLASS[d.slug]}` : '(extra)';
    console.log(`  ${d.slug.padEnd(40)} ${tag}`);
  }

  // Lookup Supabase diseases
  let supabaseDiseaseMap = new Map();
  if (supabase) {
    const { data, error } = await supabase.from('diseases').select('id, name');
    if (error) {
      console.error('\nErro ao buscar doencas no Supabase:', error.message);
    } else {
      supabaseDiseaseMap = new Map(data.map((d) => [d.name, d.id]));
      console.log(`\nDoencas no Supabase: ${data.map((d) => d.name).join(', ')}`);
    }
  }

  // Download each disease
  let totalDownloaded = 0;
  let totalRegistered = 0;
  let totalImages = 0;
  let totalErrors = 0;

  for (const disease of diseases) {
    const result = await downloadDisease(disease, session, supabaseDiseaseMap);
    totalDownloaded += result.downloaded;
    totalRegistered += result.registered;
    totalImages += result.total;
    totalErrors += result.errors;

    await sleep(500);
  }

  console.log('\n========================================');
  console.log('  RESUMO');
  console.log('========================================');
  console.log(`  Doencas processadas:    ${diseases.length}`);
  console.log(`  Imagens no site:        ${totalImages}`);
  console.log(`  Novas baixadas:         ${totalDownloaded}`);
  console.log(`  Erros de download:      ${totalErrors}`);
  console.log(`  Registradas Supabase:   ${totalRegistered}`);
  console.log(`  Diretorio: data/images/`);
  console.log('========================================\n');
}

main().catch((err) => {
  console.error('Erro fatal:', err);
  process.exit(1);
});
