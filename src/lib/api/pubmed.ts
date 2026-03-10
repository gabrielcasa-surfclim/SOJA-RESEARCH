import type { Article } from '../../types';

const BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils';

export async function searchPubMed(
  query: string,
  maxResults = 20
): Promise<Article[]> {
  try {
    // Step 1: Search for article IDs
    const searchUrl = `${BASE_URL}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(query)}&retmax=${maxResults}&retmode=json`;
    const searchResponse = await fetch(searchUrl);

    if (!searchResponse.ok) {
      console.error('PubMed esearch failed:', searchResponse.statusText);
      return [];
    }

    const searchData = await searchResponse.json();
    const ids: string[] = searchData.esearchresult?.idlist ?? [];

    if (ids.length === 0) {
      return [];
    }

    // Step 2: Get article summaries
    const summaryUrl = `${BASE_URL}/esummary.fcgi?db=pubmed&id=${ids.join(',')}&retmode=json`;
    const summaryResponse = await fetch(summaryUrl);

    if (!summaryResponse.ok) {
      console.error('PubMed esummary failed:', summaryResponse.statusText);
      return [];
    }

    const summaryData = await summaryResponse.json();
    const results = summaryData.result;

    if (!results) {
      return [];
    }

    const articles: Article[] = ids
      .filter((id) => results[id])
      .map((id) => {
        const item = results[id];
        const authors: string[] = (item.authors ?? []).map(
          (a: { name: string }) => a.name
        );
        const doi = (item.articleids ?? []).find(
          (aid: { idtype: string; value: string }) => aid.idtype === 'doi'
        )?.value;

        return {
          id: `pubmed-${id}`,
          title: item.title ?? '',
          authors,
          abstract: '',
          doi: doi ?? undefined,
          source: 'pubmed' as const,
          url: doi
            ? `https://doi.org/${doi}`
            : `https://pubmed.ncbi.nlm.nih.gov/${id}/`,
          publishedDate: item.pubdate ?? undefined,
          citationCount: undefined,
          createdAt: new Date().toISOString(),
        };
      });

    return articles;
  } catch (error) {
    console.error('PubMed search error:', error);
    return [];
  }
}
