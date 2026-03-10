import type { SearchResult } from '../../types';
import { searchPubMed } from './pubmed';
import { searchCrossRef } from './crossref';
import { searchSemanticScholar } from './semanticScholar';

export { searchPubMed } from './pubmed';
export { searchCrossRef } from './crossref';
export { searchSemanticScholar } from './semanticScholar';

export async function searchAllSources(
  query: string
): Promise<SearchResult[]> {
  const results = await Promise.allSettled([
    searchPubMed(query),
    searchCrossRef(query),
    searchSemanticScholar(query),
  ]);

  const sources: Array<SearchResult['source']> = [
    'pubmed',
    'crossref',
    'semanticscholar',
  ];

  const searchResults: SearchResult[] = results.map((result, index) => {
    const articles =
      result.status === 'fulfilled' ? result.value : [];

    if (result.status === 'rejected') {
      console.error(`Search failed for ${sources[index]}:`, result.reason);
    }

    return {
      source: sources[index],
      articles,
      total: articles.length,
    };
  });

  return searchResults;
}
