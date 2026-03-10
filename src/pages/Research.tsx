import { useState } from 'react';
import { BookOpen, Database, Loader2 } from 'lucide-react';
import { diseases } from '../data/diseases';
import type { Article, SearchResult } from '../types';
import { searchAllSources } from '../lib/api';
import SearchBar from '../components/SearchBar';
import ArticleCard from '../components/ArticleCard';

type SourceFilter = 'all' | 'pubmed' | 'crossref' | 'semanticscholar';

const sourceFilters: { id: SourceFilter; label: string }[] = [
  { id: 'all', label: 'Todas' },
  { id: 'pubmed', label: 'PubMed' },
  { id: 'crossref', label: 'CrossRef' },
  { id: 'semanticscholar', label: 'Semantic Scholar' },
];

const suggestedSearches = diseases
  .filter((d) => d.id !== 'folha-saudavel')
  .map((d) => ({
    disease: d.name,
    terms: [
      `${d.scientificName} soybean detection`,
      `${d.name} deep learning classification`,
      `soybean ${d.scientificName} image dataset`,
    ],
  }));

export default function Research() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [activeFilter, setActiveFilter] = useState<SourceFilter>('all');

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setHasSearched(true);
    try {
      const data = await searchAllSources(query.trim());
      setResults(data);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (term: string) => {
    setQuery(term);
  };

  const allArticles: Article[] = results.flatMap((r) => r.articles);

  const filteredArticles =
    activeFilter === 'all'
      ? allArticles
      : allArticles.filter((a) => a.source === activeFilter);

  const countBySource = (source: Article['source']) =>
    results.find((r) => r.source === source)?.total ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text">Pesquisa Científica</h1>
        <p className="mt-1 text-text-secondary">
          Busque artigos sobre doenças da soja em múltiplas bases de dados.
        </p>
      </div>

      {/* Search */}
      <SearchBar
        value={query}
        onChange={setQuery}
        onSearch={handleSearch}
        placeholder="Ex: Phakopsora pachyrhizi deep learning detection..."
        loading={loading}
      />

      {/* Source Filter Tabs */}
      {hasSearched && !loading && (
        <div className="flex flex-wrap gap-2">
          {sourceFilters.map((filter) => {
            const count =
              filter.id === 'all'
                ? allArticles.length
                : countBySource(filter.id as Article['source']);
            return (
              <button
                key={filter.id}
                onClick={() => setActiveFilter(filter.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeFilter === filter.id
                    ? 'bg-primary text-bg'
                    : 'bg-surface text-text-secondary hover:text-text hover:bg-surface-light'
                }`}
              >
                {filter.label}
                <span className="ml-2 px-1.5 py-0.5 rounded-full bg-bg/20 text-xs">
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-16 gap-4">
          <Loader2 size={32} className="animate-spin text-primary" />
          <p className="text-text-secondary">Buscando em todas as fontes...</p>
        </div>
      )}

      {/* Results */}
      {hasSearched && !loading && filteredArticles.length > 0 && (
        <div className="space-y-4">
          <p className="text-sm text-text-secondary">
            {filteredArticles.length} resultado
            {filteredArticles.length !== 1 ? 's' : ''} encontrado
            {filteredArticles.length !== 1 ? 's' : ''}
          </p>
          {filteredArticles.map((article) => (
            <ArticleCard key={article.id} article={article} />
          ))}
        </div>
      )}

      {/* Empty Results */}
      {hasSearched && !loading && allArticles.length === 0 && (
        <div className="bg-surface rounded-xl p-12 text-center">
          <Database size={48} className="mx-auto text-text-secondary/30" />
          <p className="mt-4 text-text-secondary">
            Nenhum resultado encontrado para &ldquo;{query}&rdquo;.
          </p>
          <p className="mt-1 text-sm text-text-secondary">
            Tente termos diferentes ou em inglês para melhores resultados.
          </p>
        </div>
      )}

      {/* Default State: Suggested Searches */}
      {!hasSearched && (
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <BookOpen size={18} className="text-primary" />
            <h2 className="text-lg font-semibold text-text">
              Sugestões de Busca
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {suggestedSearches.map((item) => (
              <div
                key={item.disease}
                className="bg-surface rounded-xl p-5 space-y-3"
              >
                <h3 className="font-semibold text-text">{item.disease}</h3>
                <div className="space-y-2">
                  {item.terms.map((term) => (
                    <button
                      key={term}
                      onClick={() => handleSuggestionClick(term)}
                      className="block w-full text-left text-sm text-primary hover:text-primary-light transition-colors truncate"
                    >
                      &ldquo;{term}&rdquo;
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
