import type { Article } from '../../types';

const BASE_URL = 'https://api.crossref.org/works';

export async function searchCrossRef(
  query: string,
  maxResults = 20
): Promise<Article[]> {
  try {
    const url = `${BASE_URL}?query=${encodeURIComponent(query)}&rows=${maxResults}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error('CrossRef search failed:', response.statusText);
      return [];
    }

    const data = await response.json();
    const items = data.message?.items ?? [];

    const articles: Article[] = items.map(
      (item: {
        DOI?: string;
        title?: string[];
        author?: { given?: string; family?: string }[];
        abstract?: string;
        published?: { 'date-parts'?: number[][] };
        'is-referenced-by-count'?: number;
      }) => {
        const authors: string[] = (item.author ?? []).map((a) =>
          [a.given, a.family].filter(Boolean).join(' ')
        );

        const dateParts = item.published?.['date-parts']?.[0];
        const publishedDate = dateParts
          ? dateParts.map((p) => String(p).padStart(2, '0')).join('-')
          : undefined;

        const doi = item.DOI ?? undefined;

        return {
          id: `crossref-${doi ?? crypto.randomUUID()}`,
          title: item.title?.[0] ?? '',
          authors,
          abstract: item.abstract ?? '',
          doi,
          source: 'crossref' as const,
          url: doi ? `https://doi.org/${doi}` : undefined,
          publishedDate,
          citationCount: item['is-referenced-by-count'] ?? undefined,
          createdAt: new Date().toISOString(),
        };
      }
    );

    return articles;
  } catch (error) {
    console.error('CrossRef search error:', error);
    return [];
  }
}
