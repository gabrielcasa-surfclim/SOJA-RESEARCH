/**
 * download-srin.js — Baixa fotos de doenças de soja do soybeanresearchinfo.com (SRIN)
 *
 * Acessa cada página de doença, extrai imagens do wp-content/uploads/,
 * e baixa as versões de maior resolução.
 *
 * Uso:
 *   node scripts/download-srin.js [--all | --doencas slug1,slug2,...]
 *
 * Exemplos:
 *   node scripts/download-srin.js --all
 *   node scripts/download-srin.js --doencas anthracnose,frogeye-leaf-spot,powdery-mildew
 */

import { createClient } from '@supabase/supabase-js';
import {
  readFileSync,
  mkdirSync,
  existsSync,
  createWriteStream,
  readdirSync,
} from 'fs';
import { resolve, dirname, join, basename } from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from 'stream/promises';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const DATA_DIR = join(ROOT, 'data', 'images');

const BASE_URL = 'https://soybeanresearchinfo.com/soybean-disease';
const CONCURRENCY = 3;

// ---------------------------------------------------------------------------
// 20 doenças do site + mapeamento pro nosso sistema
// ---------------------------------------------------------------------------
const DISEASES = [
  { slug: 'anthracnose', diseaseName: 'Antracnose' },
  { slug: 'bacterial-blight', diseaseName: null },
  { slug: 'bacterial-pustule', diseaseName: null },
  { slug: 'brown-stem-rot', diseaseName: null },
  { slug: 'cercospora-leaf-blight', diseaseName: 'Cercospora' },
  { slug: 'charcoal-rot', diseaseName: null },
  { slug: 'downy-mildew', diseaseName: null },
  { slug: 'frogeye-leaf-spot', diseaseName: 'Mancha-olho-de-rã' },
  { slug: 'green-stem-disorder', diseaseName: null },
  { slug: 'iron-deficiency-chlorosis', diseaseName: null },
  { slug: 'phytophthora-root-stem-rot', diseaseName: null },
  { slug: 'powdery-mildew', diseaseName: 'Oídio' },
  { slug: 'seedling-diseases', diseaseName: null },
  { slug: 'septoria-brown-spot', diseaseName: null },
  { slug: 'soybean-cyst-nematode-scn', diseaseName: null },
  { slug: 'soybean-vein-necrosis-virus', diseaseName: null },
  { slug: 'stem-canker', diseaseName: null },
  { slug: 'sudden-death-syndrome', diseaseName: null },
  { slug: 'viruses', diseaseName: 'Mosaico' },
  { slug: 'white-mold', diseaseName: null },
];

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
  return `srin_${slug.replace(/-/g, '_')}`;
}

// ---------------------------------------------------------------------------
// Extract image URLs from disease page HTML
// ---------------------------------------------------------------------------
function extractImageUrls(html) {
  const urls = new Set();

  // Match all image URLs from wp-content/uploads in img src and a href
  const patterns = [
    /(?:src|href)="(https?:\/\/soybeanresearchinfo\.com\/wp-content\/uploads\/[^"]+\.(?:jpg|jpeg|png|webp))"/gi,
    /(?:src|href)='(https?:\/\/soybeanresearchinfo\.com\/wp-content\/uploads\/[^']+\.(?:jpg|jpeg|png|webp))'/gi,
  ];

  for (const regex of patterns) {
    let match;
    while ((match = regex.exec(html)) !== null) {
      urls.add(match[1]);
    }
  }

  // For each image, try to find the largest version
  // WordPress generates variants like: image-1300x867.jpg, image-543x362.jpg
  // The original (without dimensions) is the largest
  const bestUrls = new Map(); // base -> { url, width }

  for (const url of urls) {
    // Extract base name without dimension suffix
    const dimMatch = url.match(/^(.+?)-(\d+)x(\d+)\.(jpg|jpeg|png|webp)$/i);
    if (dimMatch) {
      const [, base, w, , ext] = dimMatch;
      const width = parseInt(w);
      const key = base;
      const existing = bestUrls.get(key);
      if (!existing || width > existing.width) {
        bestUrls.set(key, { url, width });
      }
    } else {
      // No dimensions — could be the original or a cropped version
      const baseMatch = url.match(/^(.+)\.(jpg|jpeg|png|webp)$/i);
      if (baseMatch) {
        const key = baseMatch[1];
        // Original is always preferred (width=99999)
        bestUrls.set(key, { url, width: 99999 });
      }
    }
  }

  // Filter out tiny images (icons, logos, template thumbnails)
  return [...bestUrls.values()]
    .filter((v) => {
      if (v.width < 500) return false;
      // Exclude WordPress template images that appear on every page
      const fname = v.url.toLowerCase();
      if (fname.includes('80x80')) return false;
      if (fname.includes('silent-yield-stealer')) return false;
      if (fname.includes('aspect-ratio-16-9')) return false;
      return true;
    })
    .map((v) => v.url);
}

// ---------------------------------------------------------------------------
// Download image file
// ---------------------------------------------------------------------------
async function downloadFile(url, destPath) {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'soja-research-downloader/1.0' },
  });
  if (!res.ok) throw new Error(`${res.status}`);
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

  const { data: existing } = await supabase
    .from('disease_images')
    .select('url')
    .eq('disease_id', diseaseId)
    .eq('source', 'srin');

  const existingUrls = new Set((existing || []).map((e) => e.url));

  const newRecords = files
    .map((f) => {
      const localPath = `data/images/${localFolder}/${f}`;
      if (existingUrls.has(localPath)) return null;
      return {
        disease_id: diseaseId,
        url: localPath,
        source: 'srin',
        validated: false,
        metadata: { original_name: f, dataset: 'soybeanresearchinfo.com' },
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
async function downloadDisease(disease, supabaseDiseaseMap) {
  const { slug, diseaseName } = disease;
  const localFolder = slugToFolder(slug);
  const localDir = join(DATA_DIR, localFolder);
  const mapped = diseaseName ? `-> ${diseaseName}` : '(extra)';

  mkdirSync(localDir, { recursive: true });

  console.log(`\n--- ${slug} ${mapped} ---`);

  // Fetch page
  let html;
  try {
    const res = await fetch(`${BASE_URL}/${slug}/`, {
      headers: { 'User-Agent': 'soja-research-downloader/1.0' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    html = await res.text();
  } catch (err) {
    console.error(`  Erro ao acessar pagina: ${err.message}`);
    return { downloaded: 0, registered: 0, total: 0 };
  }

  // Extract image URLs
  const imageUrls = extractImageUrls(html);
  console.log(`  ${imageUrls.length} imagens encontradas.`);

  if (imageUrls.length === 0) {
    return { downloaded: 0, registered: 0, total: 0 };
  }

  // Build download list
  const toDownload = [];
  for (const url of imageUrls) {
    // Use the original filename from the URL
    const filename = decodeURIComponent(basename(url));
    const destPath = join(localDir, filename);
    if (existsSync(destPath)) continue;
    toDownload.push({ url, destPath, filename });
  }

  if (toDownload.length === 0) {
    console.log('  Todas ja baixadas.');
  } else {
    console.log(`  ${toDownload.length} novas para baixar.`);

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
      await sleep(300);
    });

    if (errors > 0) console.log(`  ${errors} erros de download.`);
  }

  // Register in Supabase
  const diseaseId = diseaseName ? supabaseDiseaseMap.get(diseaseName) : null;
  let registered = 0;
  if (diseaseId) {
    console.log('  Registrando no Supabase...');
    registered = await registerImages(localDir, localFolder, diseaseId);
    console.log(`  ${registered} novas registradas.`);
  }

  const totalFiles = readdirSync(localDir).filter((f) => /\.(jpg|jpeg|png|webp)$/i.test(f)).length;
  console.log(`  Total na pasta: ${totalFiles} imagens.`);

  return { downloaded: toDownload.length, registered, total: imageUrls.length };
}

// ---------------------------------------------------------------------------
// CLI & Main
// ---------------------------------------------------------------------------
async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Uso: node scripts/download-srin.js [opcoes]

Opcoes:
  --all                              Baixar todas as 20 doencas
  --doencas slug1,slug2,...          Baixar apenas doencas especificas

Doencas disponiveis:
${DISEASES.map((d) => `  ${d.slug.padEnd(35)} ${d.diseaseName ? `-> ${d.diseaseName}` : '(extra)'}`).join('\n')}

Exemplos:
  node scripts/download-srin.js --all
  node scripts/download-srin.js --doencas anthracnose,frogeye-leaf-spot,powdery-mildew
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

  const diseases = doAll
    ? DISEASES
    : DISEASES.filter((d) => selectedSlugs.has(d.slug));

  if (diseases.length === 0) {
    console.error('Nenhuma doenca selecionada.');
    process.exit(1);
  }

  console.log('========================================');
  console.log('  SRIN — Soybean Research Info');
  console.log('  soybeanresearchinfo.com');
  console.log('========================================\n');

  const mapped = diseases.filter((d) => d.diseaseName);
  const extra = diseases.filter((d) => !d.diseaseName);
  console.log(`Mapeadas: ${mapped.length} | Extras: ${extra.length} | Total: ${diseases.length}\n`);

  for (const d of diseases) {
    const tag = d.diseaseName ? `-> ${d.diseaseName}` : '(extra)';
    console.log(`  ${d.slug.padEnd(35)} ${tag}`);
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

  for (const disease of diseases) {
    const result = await downloadDisease(disease, supabaseDiseaseMap);
    totalDownloaded += result.downloaded;
    totalRegistered += result.registered;
    totalImages += result.total;

    await sleep(500);
  }

  console.log('\n========================================');
  console.log('  RESUMO');
  console.log('========================================');
  console.log(`  Doencas processadas:    ${diseases.length}`);
  console.log(`  Imagens no site:        ${totalImages}`);
  console.log(`  Novas baixadas:         ${totalDownloaded}`);
  console.log(`  Registradas Supabase:   ${totalRegistered}`);
  console.log(`  Diretorio: data/images/`);
  console.log('========================================\n');
}

main().catch((err) => {
  console.error('Erro fatal:', err);
  process.exit(1);
});
