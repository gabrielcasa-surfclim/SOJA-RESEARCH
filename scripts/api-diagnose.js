/**
 * api-diagnose.js — API REST local para diagnóstico de doenças em soja.
 *
 * Spawna predict.py em modo --server (stdin/stdout JSON) para manter o modelo
 * carregado em memória. Cada request envia base64 via stdin, recebe JSON via stdout.
 *
 * Endpoints:
 *   POST /diagnose        { image: "base64..." }  → diagnóstico + knowledge
 *   POST /diagnose/url    { url: "https://..." }   → baixa imagem e diagnostica
 *   GET  /health                                    → status do modelo
 *
 * Uso:
 *   node scripts/api-diagnose.js [--port 3001]
 *
 * Requer: python3 com torch, modelo treinado em training/best_model.pth
 */

import { createClient } from '@supabase/supabase-js';
import { readFileSync, existsSync, writeFileSync, unlinkSync } from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { spawn, execSync } from 'child_process';
import { createServer } from 'http';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const TRAINING_DIR = join(ROOT, 'training');

const args = process.argv.slice(2);
const portIdx = args.indexOf('--port');
const PORT = portIdx !== -1 ? parseInt(args[portIdx + 1]) : 3001;

// ---------------------------------------------------------------------------
// Supabase (optional — for disease knowledge lookup)
// ---------------------------------------------------------------------------
let supabase = null;
const envPath = resolve(ROOT, '.env');

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

  const url = envVars.VITE_SUPABASE_URL;
  const key = envVars.VITE_SUPABASE_ANON_KEY;
  if (url && key) supabase = createClient(url, key);
}

// ---------------------------------------------------------------------------
// Disease knowledge cache (from Supabase diseases table)
// ---------------------------------------------------------------------------
let diseaseKnowledge = new Map(); // name -> { scientific_name, symptoms, ... }

async function loadDiseaseKnowledge() {
  if (!supabase) return;
  const { data, error } = await supabase
    .from('diseases')
    .select('name, scientific_name, symptoms, severity, conditions, management, fungicides');
  if (error) {
    console.error('Erro ao carregar knowledge:', error.message);
    return;
  }
  for (const d of data) {
    diseaseKnowledge.set(d.name, {
      scientific_name: d.scientific_name,
      symptoms: d.symptoms,
      severity: d.severity,
      conditions: d.conditions,
      management: d.management,
      fungicides: d.fungicides,
    });
  }
  console.log(`Knowledge carregado: ${diseaseKnowledge.size} doencas`);
}

// Map model class names -> Supabase disease names
const CLASS_TO_DISEASE = {
  'Ferrugem': 'Ferrugem Asiática',
  'Frogeye': 'Mancha-olho-de-rã',
  'Mancha-alvo': 'Mancha-alvo',
  'Mosaico': 'Mosaico',
  'Oídio': 'Oídio',
  'Saudável': 'Folha Saudável',
  'Target Spot': 'Mancha-alvo',
  'Antracnose': 'Antracnose',
  'Cercospora': 'Mancha-olho-de-rã',
};

function lookupKnowledge(className) {
  const diseaseName = CLASS_TO_DISEASE[className] || className;
  return diseaseKnowledge.get(diseaseName) || null;
}

// ---------------------------------------------------------------------------
// Python predict.py subprocess (persistent, stdin/stdout JSON)
// ---------------------------------------------------------------------------
let pythonProcess = null;
let pythonReady = false;
let modelInfo = {};
let pendingResolve = null;
let outputBuffer = '';

function startPython() {
  return new Promise((resolve, reject) => {
    console.log('Iniciando predict.py --server...');

    pythonProcess = spawn('python3', ['predict.py', '--server'], {
      cwd: TRAINING_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    pythonProcess.stderr.on('data', (data) => {
      // PyTorch warnings go to stderr
      const msg = data.toString().trim();
      if (msg && !msg.includes('UserWarning')) {
        console.error('[python]', msg);
      }
    });

    pythonProcess.on('close', (code) => {
      console.error(`predict.py encerrou com codigo ${code}`);
      pythonReady = false;
      pythonProcess = null;
      if (pendingResolve) {
        pendingResolve.reject(new Error('Python process died'));
        pendingResolve = null;
      }
    });

    // First line is the ready status
    let firstLine = true;
    pythonProcess.stdout.on('data', (data) => {
      outputBuffer += data.toString();
      const lines = outputBuffer.split('\n');
      outputBuffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line);
          if (firstLine) {
            firstLine = false;
            if (parsed.status === 'ready') {
              pythonReady = true;
              modelInfo = parsed;
              console.log(`Modelo pronto: ${parsed.classes.length} classes, acc: ${(parsed.accuracy * 100).toFixed(1)}%`);
              resolve();
            } else {
              reject(new Error(`Python nao iniciou: ${line}`));
            }
          } else if (pendingResolve) {
            const { resolve: res } = pendingResolve;
            pendingResolve = null;
            res(parsed);
          }
        } catch {
          console.error('[python stdout]', line);
        }
      }
    });

    // Timeout
    setTimeout(() => {
      if (!pythonReady) reject(new Error('Timeout ao iniciar predict.py'));
    }, 30000);
  });
}

function classifyBase64(base64Image) {
  return new Promise((resolve, reject) => {
    if (!pythonReady || !pythonProcess) {
      return reject(new Error('Modelo nao esta pronto'));
    }
    if (pendingResolve) {
      return reject(new Error('Classificacao ja em andamento'));
    }

    pendingResolve = { resolve, reject };

    const timeout = setTimeout(() => {
      if (pendingResolve) {
        pendingResolve = null;
        reject(new Error('Timeout na classificacao (30s)'));
      }
    }, 30000);

    const origResolve = pendingResolve.resolve;
    pendingResolve.resolve = (val) => {
      clearTimeout(timeout);
      origResolve(val);
    };

    pythonProcess.stdin.write(JSON.stringify({ image: base64Image }) + '\n');
  });
}

// ---------------------------------------------------------------------------
// HTTP Server
// ---------------------------------------------------------------------------
function parseBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString()));
      } catch {
        reject(new Error('JSON invalido'));
      }
    });
    req.on('error', reject);
  });
}

function sendJson(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  });
  res.end(JSON.stringify(data));
}

async function handleDiagnose(req, res) {
  let body;
  try {
    body = await parseBody(req);
  } catch {
    return sendJson(res, 400, { error: 'JSON invalido no body' });
  }

  if (!body.image) {
    return sendJson(res, 400, { error: 'Campo "image" (base64) obrigatorio' });
  }

  try {
    const result = await classifyBase64(body.image);
    if (result.error) {
      return sendJson(res, 500, { error: result.error });
    }

    // Enrich with disease knowledge
    const knowledge = lookupKnowledge(result.disease);

    return sendJson(res, 200, {
      disease: result.disease,
      confidence: result.confidence,
      top3: result.top3.map((t) => ({
        ...t,
        knowledge: lookupKnowledge(t.class),
      })),
      knowledge,
      model_accuracy: result.model_accuracy,
    });
  } catch (err) {
    return sendJson(res, 500, { error: err.message });
  }
}

async function handleDiagnoseUrl(req, res) {
  let body;
  try {
    body = await parseBody(req);
  } catch {
    return sendJson(res, 400, { error: 'JSON invalido no body' });
  }

  if (!body.url) {
    return sendJson(res, 400, { error: 'Campo "url" obrigatorio' });
  }

  try {
    // Download image and convert to base64
    const tmpPath = join(tmpdir(), `soja-${randomUUID()}.jpg`);
    execSync(`curl -sL -o "${tmpPath}" "${body.url}"`, { timeout: 15000 });
    const imageBytes = readFileSync(tmpPath);
    unlinkSync(tmpPath);

    const base64Image = imageBytes.toString('base64');
    const result = await classifyBase64(base64Image);
    if (result.error) {
      return sendJson(res, 500, { error: result.error });
    }

    const knowledge = lookupKnowledge(result.disease);

    return sendJson(res, 200, {
      disease: result.disease,
      confidence: result.confidence,
      top3: result.top3.map((t) => ({
        ...t,
        knowledge: lookupKnowledge(t.class),
      })),
      knowledge,
      model_accuracy: result.model_accuracy,
      source_url: body.url,
    });
  } catch (err) {
    return sendJson(res, 500, { error: err.message });
  }
}

const server = createServer(async (req, res) => {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    return sendJson(res, 204, {});
  }

  const url = req.url.split('?')[0];

  if (req.method === 'GET' && url === '/health') {
    return sendJson(res, 200, {
      status: pythonReady ? 'ok' : 'loading',
      model: modelInfo,
      knowledge_loaded: diseaseKnowledge.size,
    });
  }

  if (req.method === 'POST' && url === '/diagnose') {
    return handleDiagnose(req, res);
  }

  if (req.method === 'POST' && url === '/diagnose/url') {
    return handleDiagnoseUrl(req, res);
  }

  return sendJson(res, 404, { error: 'Not found. Endpoints: POST /diagnose, POST /diagnose/url, GET /health' });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
async function main() {
  console.log('========================================');
  console.log('  Soja Research — API de Diagnostico');
  console.log('========================================\n');

  await loadDiseaseKnowledge();
  await startPython();

  server.listen(PORT, () => {
    console.log(`\nAPI rodando em http://localhost:${PORT}`);
    console.log(`\nEndpoints:`);
    console.log(`  POST /diagnose      { image: "base64..." }`);
    console.log(`  POST /diagnose/url  { url: "https://..." }`);
    console.log(`  GET  /health\n`);
    console.log(`Teste rapido:`);
    console.log(`  curl -s localhost:${PORT}/health | jq`);
    console.log(`  curl -s -X POST localhost:${PORT}/diagnose/url -H 'Content-Type: application/json' -d '{"url":"https://www.doencasdeplantas.com.br/img/uploads/plants_or_seeds/soja-42/oidio/soja_oidio%20(8).JPG"}' | jq`);
  });
}

main().catch((err) => {
  console.error('Erro fatal:', err.message);
  process.exit(1);
});
