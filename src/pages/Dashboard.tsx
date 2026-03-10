import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bug, Camera, FileText, Target, CheckSquare, Square, Search, Loader2, ExternalLink } from 'lucide-react';
import { diseases } from '../data/diseases';
import type { Disease, Article } from '../types';
import { searchAllSources } from '../lib/api';
import StatsCard from '../components/StatsCard';
import DiseaseCard from '../components/DiseaseCard';

const totalImages = diseases.reduce((sum, d) => sum + d.imageCount, 0);
const totalArticles = diseases.reduce((sum, d) => sum + d.articleCount, 0);
const maxImages = Math.max(...diseases.map((d) => d.imageCount));

const diseaseSearchTerms: Record<string, string> = {
  'ferrugem-asiatica': '"soybean rust" OR "Phakopsora pachyrhizi" AND "detection" AND "deep learning"',
  'mancha-olho-de-ra': '"soybean frogeye leaf spot" OR "Cercospora sojina" AND "image classification"',
  'oidio': '"soybean powdery mildew" OR "Erysiphe diffusa" AND "image classification"',
  'antracnose': '"soybean anthracnose" OR "Colletotrichum truncatum" AND "detection"',
  'mosaico': '"soybean mosaic virus" OR "SMV" AND "detection" AND "deep learning"',
  'mancha-alvo': '"target spot soybean" OR "Corynespora cassiicola" AND "computer vision"',
};

const checklist = [
  { label: 'Configurar plataforma', done: true },
  { label: 'Mapear doenças-alvo', done: true },
  { label: 'Buscar artigos científicos', done: false },
  { label: 'Coletar dataset inicial', done: false },
  { label: 'Treinar modelo v1', done: false },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [fetchingArticles, setFetchingArticles] = useState(false);
  const [fetchProgress, setFetchProgress] = useState('');
  const [fetchedArticles, setFetchedArticles] = useState<{ disease: string; articles: Article[] }[]>([]);

  const handleDiseaseClick = (disease: Disease) => {
    navigate(`/doenca/${disease.id}`);
  };

  const handleFetchAll = async () => {
    setFetchingArticles(true);
    setFetchedArticles([]);
    const results: { disease: string; articles: Article[] }[] = [];

    const entries = Object.entries(diseaseSearchTerms);
    for (let i = 0; i < entries.length; i++) {
      const [diseaseId, query] = entries[i];
      const disease = diseases.find((d) => d.id === diseaseId);
      setFetchProgress(`Buscando ${disease?.name ?? diseaseId}... (${i + 1}/${entries.length})`);

      try {
        const searchResults = await searchAllSources(query);
        const allArticles = searchResults.flatMap((r) => r.articles);
        results.push({ disease: disease?.name ?? diseaseId, articles: allArticles });
      } catch {
        results.push({ disease: disease?.name ?? diseaseId, articles: [] });
      }

      // Small delay between diseases to avoid rate limiting
      if (i < entries.length - 1) {
        await new Promise((r) => setTimeout(r, 800));
      }
    }

    setFetchedArticles(results);
    setFetchProgress('');
    setFetchingArticles(false);
  };

  return (
    <div className="space-y-8">
      {/* Stats Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Total Doenças"
          value={7}
          icon={Bug}
          color="bg-primary/15 text-primary"
        />
        <StatsCard
          title="Total Imagens"
          value={totalImages.toLocaleString('pt-BR')}
          icon={Camera}
          color="bg-accent/15 text-accent"
        />
        <StatsCard
          title="Total Artigos"
          value={totalArticles}
          icon={FileText}
          color="bg-success/15 text-success"
        />
        <StatsCard
          title="Acurácia Alvo"
          value="96.8%"
          icon={Target}
          color="bg-warning/15 text-warning"
        />
      </div>

      {/* Doenças Monitoradas */}
      <section>
        <h2 className="text-xl font-bold text-text mb-4">
          Doenças Monitoradas
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {diseases.map((disease) => (
            <DiseaseCard
              key={disease.id}
              disease={disease}
              onClick={handleDiseaseClick}
            />
          ))}
        </div>
      </section>

      {/* Progresso do Dataset */}
      <section>
        <h2 className="text-xl font-bold text-text mb-4">
          Progresso do Dataset
        </h2>
        <div className="bg-surface rounded-xl p-6 space-y-4">
          {diseases.map((disease) => (
            <div key={disease.id} className="flex items-center gap-4">
              <span className="text-lg">{disease.icon}</span>
              <span className="w-36 text-sm text-text truncate">
                {disease.name}
              </span>
              <div className="flex-1 bg-surface-light rounded-full h-4 overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all"
                  style={{
                    width: `${(disease.imageCount / maxImages) * 100}%`,
                  }}
                />
              </div>
              <span className="w-16 text-right text-sm text-text-secondary tabular-nums">
                {disease.imageCount}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Buscar Artigos */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-text">Artigos Científicos</h2>
          <button
            onClick={handleFetchAll}
            disabled={fetchingArticles}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-bg rounded-lg font-medium hover:bg-primary-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {fetchingArticles ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Search size={16} />
            )}
            {fetchingArticles ? 'Buscando...' : 'Buscar todos'}
          </button>
        </div>

        {fetchProgress && (
          <div className="bg-surface rounded-xl p-4 mb-4">
            <p className="text-sm text-primary flex items-center gap-2">
              <Loader2 size={14} className="animate-spin" />
              {fetchProgress}
            </p>
          </div>
        )}

        {fetchedArticles.length > 0 && (
          <div className="bg-surface rounded-xl p-6 space-y-5">
            <p className="text-sm text-text-secondary">
              {fetchedArticles.reduce((sum, r) => sum + r.articles.length, 0)} artigos encontrados em {fetchedArticles.length} doenças
            </p>
            {fetchedArticles.map((result) => (
              <div key={result.disease}>
                <h3 className="text-sm font-semibold text-text mb-2">
                  {result.disease}
                  <span className="ml-2 text-text-secondary font-normal">
                    ({result.articles.length} artigos)
                  </span>
                </h3>
                {result.articles.length > 0 ? (
                  <div className="space-y-2">
                    {result.articles.slice(0, 5).map((article) => (
                      <div
                        key={article.id}
                        className="flex items-start gap-3 p-3 bg-surface-light rounded-lg"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-text truncate">{article.title}</p>
                          <p className="text-xs text-text-secondary mt-1">
                            {article.authors.slice(0, 3).join(', ')}
                            {article.authors.length > 3 ? ' et al.' : ''}
                            {article.publishedDate ? ` · ${article.publishedDate}` : ''}
                            <span className="ml-2 px-1.5 py-0.5 rounded bg-bg text-xs">
                              {article.source}
                            </span>
                          </p>
                        </div>
                        {article.url && (
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-primary hover:text-primary-light shrink-0 mt-0.5"
                          >
                            <ExternalLink size={14} />
                          </a>
                        )}
                      </div>
                    ))}
                    {result.articles.length > 5 && (
                      <p className="text-xs text-text-secondary pl-3">
                        +{result.articles.length - 5} artigos...
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-text-secondary pl-3">
                    Nenhum artigo encontrado.
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Próximos Passos */}
      <section>
        <h2 className="text-xl font-bold text-text mb-4">Próximos Passos</h2>
        <div className="bg-surface rounded-xl p-6 space-y-3">
          {checklist.map((item) => (
            <div key={item.label} className="flex items-center gap-3">
              {item.done ? (
                <CheckSquare size={20} className="text-success shrink-0" />
              ) : (
                <Square size={20} className="text-text-secondary shrink-0" />
              )}
              <span
                className={`text-sm ${
                  item.done
                    ? 'text-text-secondary line-through'
                    : 'text-text'
                }`}
              >
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
