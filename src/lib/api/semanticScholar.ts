import type { Article } from '../../types';

const BASE_URL = import.meta.env.DEV
  ? '/api/semanticscholar/graph/v1/paper/search'
  : 'https://api.semanticscholar.org/graph/v1/paper/search';

async function fetchWithRetry(url: string, retries = 2, delay = 1500): Promise<Response> {
  const response = await fetch(url);
  if (response.status === 429 && retries > 0) {
    await new Promise((r) => setTimeout(r, delay));
    return fetchWithRetry(url, retries - 1, delay * 2);
  }
  return response;
}

export async function searchSemanticScholar(
  query: string,
  maxResults = 20
): Promise<Article[]> {
  try {
    const fields = 'title,abstract,authors,year,citationCount,externalIds,url';
    const url = `${BASE_URL}?query=${encodeURIComponent(query)}&limit=${maxResults}&fields=${fields}`;
    const response = await fetchWithRetry(url);

    if (!response.ok) {
      console.error('Semantic Scholar search failed:', response.status, response.statusText);
      return [];
    }

    const data = await response.json();
    const papers = data.data ?? [];

    const articles: Article[] = papers.map(
      (paper: {
        paperId?: string;
        title?: string;
        abstract?: string;
        authors?: { name: string }[];
        year?: number;
        citationCount?: number;
        externalIds?: { DOI?: string };
        url?: string;
      }) => {
        const authors: string[] = (paper.authors ?? []).map((a) => a.name);
        const doi = paper.externalIds?.DOI ?? undefined;

        return {
          id: `semanticscholar-${paper.paperId ?? crypto.randomUUID()}`,
          title: paper.title ?? '',
          authors,
          abstract: paper.abstract ?? '',
          doi,
          source: 'semanticscholar' as const,
          url: paper.url ?? (doi ? `https://doi.org/${doi}` : undefined),
          publishedDate: paper.year ? String(paper.year) : undefined,
          citationCount: paper.citationCount ?? undefined,
          createdAt: new Date().toISOString(),
        };
      }
    );

    return articles;
  } catch (error) {
    console.error('Semantic Scholar search error:', error);
    return [];
  }
}
