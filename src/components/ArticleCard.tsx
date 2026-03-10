import { ExternalLink, Calendar, Quote } from 'lucide-react';
import type { Article } from '../types';

interface ArticleCardProps {
  article: Article;
}

const sourceConfig: Record<Article['source'], { label: string; className: string }> = {
  pubmed: { label: 'PubMed', className: 'bg-blue-500/15 text-blue-400' },
  crossref: { label: 'Crossref', className: 'bg-orange-500/15 text-orange-400' },
  semanticscholar: {
    label: 'Semantic Scholar',
    className: 'bg-purple-500/15 text-purple-400',
  },
};

export default function ArticleCard({ article }: ArticleCardProps) {
  const source = sourceConfig[article.source];

  return (
    <div className="bg-surface rounded-xl p-5 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-lg font-semibold text-text leading-snug">
          {article.title}
        </h3>
        <span
          className={`shrink-0 px-2.5 py-0.5 rounded-full text-xs font-medium ${source.className}`}
        >
          {source.label}
        </span>
      </div>

      {/* Authors */}
      <p className="text-sm text-text-secondary">
        {article.authors.join(', ')}
      </p>

      {/* Abstract */}
      <p className="text-sm text-text-secondary line-clamp-3 leading-relaxed">
        {article.abstract}
      </p>

      {/* Methods tags */}
      {article.methods && article.methods.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {article.methods.map((method) => (
            <span
              key={method}
              className="px-2 py-0.5 rounded-md bg-surface-light text-xs text-text-secondary"
            >
              {method}
            </span>
          ))}
        </div>
      )}

      {/* Bottom row */}
      <div className="flex items-center gap-4 pt-2 border-t border-border text-sm text-text-secondary">
        {article.publishedDate && (
          <span className="flex items-center gap-1.5">
            <Calendar size={14} />
            {article.publishedDate}
          </span>
        )}

        {article.citationCount != null && (
          <span className="flex items-center gap-1.5">
            <Quote size={14} />
            {article.citationCount}
          </span>
        )}

        {article.doi && (
          <a
            href={`https://doi.org/${article.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto flex items-center gap-1.5 text-primary hover:text-primary-light transition-colors"
          >
            DOI
            <ExternalLink size={14} />
          </a>
        )}
      </div>
    </div>
  );
}
