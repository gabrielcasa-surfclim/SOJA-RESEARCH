import { createClient } from '@supabase/supabase-js';
import {
  readFileSync,
  mkdirSync,
  existsSync,
  createWriteStream,
  readdirSync,
  unlinkSync,
} from 'fs';
import { resolve, dirname, join, extname } from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from 'stream/promises';
import { execSync } from 'child_process';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const DATA_DIR = join(ROOT, 'data', 'images');
const TMP_DIR = join(ROOT, 'data', 'tmp');

const CONCURRENCY = 5;

// ---------------------------------------------------------------------------
// Sources — PlantVillage (GitHub) + ASDID (Zenodo)
// ---------------------------------------------------------------------------

const PLANTVILLAGE = {
  name: 'PlantVillage',
  source: 'plantvillage',
  repo: 'spMohanty/PlantVillage-Dataset',
  basePath: 'raw/color',
  classes: [
    {
      folder: 'Soybean___healthy',
      localName: 'healthy',
      diseaseName: 'Folha Saudável',
    },
  ],
};

// ASDID — Auburn Soybean Disease Image Dataset (Zenodo, CC0, no auth)
// https://zenodo.org/records/7304859
// ~9,981 images, 8 classes, field-captured photos
const ASDID = {
  name: 'ASDID',
  source: 'asdid',
  zenodoRecord: '7304859',
  classes: [
    {
      zipName: 'soybean_rust.zip',
      localName: 'soybean_rust',
      diseaseName: 'Ferrugem Asiática',
      sizeMB: 7988,
    },
    {
      zipName: 'target_spot.zip',
      localName: 'target_spot',
      diseaseName: 'Mancha-alvo',
      sizeMB: 4057,
    },
    {
      zipName: 'frogeye.zip',
      localName: 'frogeye',
      diseaseName: 'Mancha-olho-de-rã',
      sizeMB: 5625,
    },
    {
      zipName: 'healthy.zip',
      localName: 'healthy_asdid',
      diseaseName: 'Folha Saudável',
      sizeMB: 6776,
    },
    {
      zipName: 'bacterial_blight.zip',
      localName: 'bacterial_blight',
      diseaseName: null, // not in our 7 diseases
      sizeMB: 2716,
    },
    {
      zipName: 'cercospora_leaf_blight.zip',
      localName: 'cercospora_leaf_blight',
      diseaseName: null,
      sizeMB: 6763,
    },
    {
      zipName: 'downey_mildew.zip',
      localName: 'downey_mildew',
      diseaseName: null,
      sizeMB: 2793,
    },
    {
      zipName: 'potassium_deficiency.zip',
      localName: 'potassium_deficiency',
      diseaseName: null,
      sizeMB: 3236,
    },
  ],
};

// ---------------------------------------------------------------------------
// Env & Supabase
// ---------------------------------------------------------------------------
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
  console.error('VITE_SUPABASE_URL e VITE_SUPABASE_ANON_KEY sao obrigatorios no .env');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function githubFetch(url) {
  const headers = { 'User-Agent': 'soja-research-downloader' };
  if (process.env.GITHUB_TOKEN) {
    headers['Authorization'] = `token ${process.env.GITHUB_TOKEN}`;
  }
  const res = await fetch(url, { headers });
  if (res.status === 403) {
    const reset = res.headers.get('x-ratelimit-reset');
    const waitSec = reset ? Math.max(0, Number(reset) - Math.floor(Date.now() / 1000)) : 60;
    console.warn(`  Rate limited. Aguardando ${waitSec}s...`);
    await sleep(waitSec * 1000);
    return githubFetch(url);
  }
  if (!res.ok) throw new Error(`GitHub API ${res.status}: ${await res.text()}`);
  return res.json();
}

async function downloadFile(url, destPath) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed ${res.status}: ${url}`);
  const ws = createWriteStream(destPath);
  await pipeline(res.body, ws);
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

/** Download large file with curl (supports resume, shows progress) */
function downloadWithCurl(url, destPath) {
  console.log(`  Baixando com curl: ${destPath}`);
  execSync(`curl -L -C - --progress-bar -o "${destPath}" "${url}"`, {
    stdio: 'inherit',
  });
}

/** Register images in Supabase disease_images table */
async function registerImages(localDir, localName, diseaseId, source) {
  if (!existsSync(localDir)) return 0;

  const files = readdirSync(localDir).filter((f) =>
    /\.(jpg|jpeg|png|webp)$/i.test(f)
  );

  if (files.length === 0) return 0;

  // Get already registered
  const { data: existing } = await supabase
    .from('disease_images')
    .select('url')
    .eq('disease_id', diseaseId)
    .eq('source', source);

  const existingUrls = new Set((existing || []).map((e) => e.url));

  const newRecords = files
    .map((f) => {
      const localPath = `data/images/${localName}/${f}`;
      if (existingUrls.has(localPath)) return null;
      return {
        disease_id: diseaseId,
        url: localPath,
        source,
        validated: false,
        metadata: { original_name: f, dataset: source === 'plantvillage' ? 'PlantVillage' : 'ASDID' },
      };
    })
    .filter(Boolean);

  if (newRecords.length === 0) return 0;

  // Insert in batches
  const BATCH = 100;
  let registered = 0;
  for (let i = 0; i < newRecords.length; i += BATCH) {
    const batch = newRecords.slice(i, i + BATCH);
    const { error } = await supabase.from('disease_images').insert(batch);
    if (error) {
      console.error(`  Erro ao inserir batch ${i}: ${error.message}`);
    } else {
      registered += batch.length;
    }
  }
  return registered;
}

// ---------------------------------------------------------------------------
// PlantVillage downloader (individual images via GitHub API)
// ---------------------------------------------------------------------------
async function downloadPlantVillage(diseaseMap) {
  console.log('\n========================================');
  console.log('  PLANTVILLAGE (GitHub)');
  console.log('========================================\n');

  const { repo, basePath, classes } = PLANTVILLAGE;
  let totalDl = 0;
  let totalReg = 0;

  for (const cls of classes) {
    const diseaseId = diseaseMap.get(cls.diseaseName);
    if (!diseaseId) {
      console.warn(`  "${cls.diseaseName}" nao encontrada no Supabase. Pulando.`);
      continue;
    }

    const localDir = join(DATA_DIR, cls.localName);
    mkdirSync(localDir, { recursive: true });

    console.log(`--- ${cls.folder} -> ${cls.diseaseName} ---`);
    console.log('  Listando imagens no GitHub...');

    const apiUrl = `https://api.github.com/repos/${repo}/contents/${basePath}/${cls.folder}`;
    const files = await githubFetch(apiUrl);
    const imageFiles = files.filter(
      (f) => f.type === 'file' && /\.(jpg|jpeg|png)$/i.test(f.name)
    );
    console.log(`  ${imageFiles.length} imagens encontradas.`);

    const toDownload = imageFiles.filter(
      (f) => !existsSync(join(localDir, f.name))
    );
    console.log(`  ${toDownload.length} novas para baixar.\n`);

    let downloaded = 0;
    await asyncPool(CONCURRENCY, toDownload, async (file) => {
      const dest = join(localDir, file.name);
      const url = `https://raw.githubusercontent.com/${repo}/master/${basePath}/${cls.folder}/${encodeURIComponent(file.name)}`;
      try {
        await downloadFile(url, dest);
        downloaded++;
        if (downloaded % 50 === 0 || downloaded === toDownload.length) {
          console.log(`  Baixadas: ${downloaded}/${toDownload.length}`);
        }
      } catch (err) {
        console.error(`  Erro: ${file.name}: ${err.message}`);
      }
    });
    totalDl += downloaded;

    console.log('  Registrando no Supabase...');
    const reg = await registerImages(localDir, cls.localName, diseaseId, 'plantvillage');
    totalReg += reg;
    console.log(`  ${reg} novas registradas.\n`);
  }

  return { downloaded: totalDl, registered: totalReg };
}

// ---------------------------------------------------------------------------
// ASDID downloader (zip files from Zenodo)
// ---------------------------------------------------------------------------
async function downloadASDID(diseaseMap, selectedClasses) {
  console.log('\n========================================');
  console.log('  ASDID — Auburn Soybean Disease Dataset (Zenodo)');
  console.log('========================================\n');

  mkdirSync(TMP_DIR, { recursive: true });

  const classes = selectedClasses
    ? ASDID.classes.filter((c) => selectedClasses.includes(c.localName))
    : ASDID.classes.filter((c) => c.diseaseName !== null);

  let totalDl = 0;
  let totalReg = 0;

  for (const cls of classes) {
    if (!cls.diseaseName) {
      console.log(`  Pulando ${cls.localName} (sem doenca mapeada).\n`);
      continue;
    }

    const diseaseId = diseaseMap.get(cls.diseaseName);
    if (!diseaseId) {
      console.warn(`  "${cls.diseaseName}" nao encontrada no Supabase. Pulando.\n`);
      continue;
    }

    const localDir = join(DATA_DIR, cls.localName);

    // Check if already extracted
    if (existsSync(localDir)) {
      const existingFiles = readdirSync(localDir).filter((f) =>
        /\.(jpg|jpeg|png|webp)$/i.test(f)
      );
      if (existingFiles.length > 0) {
        console.log(`--- ${cls.localName} -> ${cls.diseaseName} ---`);
        console.log(`  Ja existem ${existingFiles.length} imagens. Pulando download.\n`);

        console.log('  Registrando no Supabase...');
        const reg = await registerImages(localDir, cls.localName, diseaseId, 'asdid');
        totalReg += reg;
        console.log(`  ${reg} novas registradas.\n`);
        continue;
      }
    }

    mkdirSync(localDir, { recursive: true });

    console.log(`--- ${cls.localName} -> ${cls.diseaseName} (~${cls.sizeMB} MB) ---`);

    const zipUrl = `https://zenodo.org/records/${ASDID.zenodoRecord}/files/${cls.zipName}?download=1`;
    const zipPath = join(TMP_DIR, cls.zipName);

    // Download zip
    if (!existsSync(zipPath)) {
      try {
        downloadWithCurl(zipUrl, zipPath);
      } catch (err) {
        console.error(`  Erro no download: ${err.message}`);
        continue;
      }
    } else {
      console.log(`  Zip ja existe: ${zipPath}`);
    }

    // Extract
    console.log(`  Extraindo para ${localDir}...`);
    try {
      execSync(`unzip -q -j -o "${zipPath}" -d "${localDir}"`, { stdio: 'inherit' });

      // Remove non-image files that might have been extracted
      for (const f of readdirSync(localDir)) {
        if (!/\.(jpg|jpeg|png|webp)$/i.test(f)) {
          unlinkSync(join(localDir, f));
        }
      }

      const extracted = readdirSync(localDir).filter((f) =>
        /\.(jpg|jpeg|png|webp)$/i.test(f)
      );
      totalDl += extracted.length;
      console.log(`  ${extracted.length} imagens extraidas.\n`);
    } catch (err) {
      console.error(`  Erro ao extrair: ${err.message}`);
      continue;
    }

    // Register in Supabase
    console.log('  Registrando no Supabase...');
    const reg = await registerImages(localDir, cls.localName, diseaseId, 'asdid');
    totalReg += reg;
    console.log(`  ${reg} novas registradas.\n`);

    // Optionally delete zip to save space
    if (process.env.KEEP_ZIPS !== '1') {
      console.log(`  Removendo zip temporario...`);
      unlinkSync(zipPath);
    }
  }

  return { downloaded: totalDl, registered: totalReg };
}

// ---------------------------------------------------------------------------
// CLI & Main
// ---------------------------------------------------------------------------
function printUsage() {
  console.log(`
Uso: node scripts/download-images.js [opcoes]

Fontes disponiveis:
  --plantvillage     Baixar Soybean___healthy do PlantVillage (GitHub, ~1000 imgs)
  --asdid            Baixar dataset ASDID do Zenodo (~9,981 imgs, 8 classes)
  --all              Baixar todas as fontes

Filtros ASDID (classes com doencas mapeadas):
  --classes rust,target,frogeye     Baixar apenas classes especificas

  Classes disponiveis:
    soybean_rust        -> Ferrugem Asiatica       (~8.0 GB)
    target_spot         -> Mancha-alvo             (~4.1 GB)
    frogeye             -> Mancha-olho-de-ra       (~5.6 GB)
    healthy_asdid       -> Folha Saudavel          (~6.8 GB)
    bacterial_blight    -> (sem mapeamento)        (~2.7 GB)
    cercospora_leaf_blight -> (sem mapeamento)     (~6.8 GB)
    downey_mildew       -> (sem mapeamento)        (~2.8 GB)
    potassium_deficiency -> (sem mapeamento)       (~3.2 GB)

Variaveis de ambiente:
  GITHUB_TOKEN=xxx     Token GitHub (evita rate limit)
  KEEP_ZIPS=1          Nao deletar zips apos extracao

Exemplos:
  node scripts/download-images.js --plantvillage
  node scripts/download-images.js --asdid --classes soybean_rust,target_spot,frogeye
  node scripts/download-images.js --all
`);
}

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    printUsage();
    process.exit(0);
  }

  const doPlantVillage = args.includes('--plantvillage') || args.includes('--all');
  const doAsdid = args.includes('--asdid') || args.includes('--all');

  if (!doPlantVillage && !doAsdid) {
    printUsage();
    process.exit(0);
  }

  // Parse --classes
  let selectedClasses = null;
  const classesIdx = args.indexOf('--classes');
  if (classesIdx !== -1 && args[classesIdx + 1]) {
    selectedClasses = args[classesIdx + 1].split(',').map((c) => c.trim());
  }

  console.log('=== Soja Research — Image Downloader ===\n');

  // Lookup diseases
  const { data: diseases, error } = await supabase.from('diseases').select('id, name');
  if (error) {
    console.error('Erro ao buscar doencas:', error.message);
    process.exit(1);
  }
  const diseaseMap = new Map(diseases.map((d) => [d.name, d.id]));
  console.log(`Doencas no banco: ${diseases.map((d) => d.name).join(', ')}`);

  let totalDl = 0;
  let totalReg = 0;

  if (doPlantVillage) {
    const r = await downloadPlantVillage(diseaseMap);
    totalDl += r.downloaded;
    totalReg += r.registered;
  }

  if (doAsdid) {
    const r = await downloadASDID(diseaseMap, selectedClasses);
    totalDl += r.downloaded;
    totalReg += r.registered;
  }

  console.log('\n=== Resumo Final ===');
  console.log(`  Imagens processadas: ${totalDl}`);
  console.log(`  Registros novos no Supabase: ${totalReg}`);
  console.log(`  Diretorio: data/images/`);
  console.log('\nDone!');
}

main().catch((err) => {
  console.error('Erro fatal:', err);
  process.exit(1);
});
