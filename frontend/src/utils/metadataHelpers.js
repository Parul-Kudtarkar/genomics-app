// Null-safe, Crossref-preferred helpers for research paper metadata
export const getBestTitle = (match) =>
  match?.metadata?.crossref_title ||
  match?.metadata?.extracted_title ||
  match?.title ||
  'Unknown Title';

export const getBestJournal = (match) =>
  match?.metadata?.crossref_journal ||
  match?.metadata?.journal ||
  'Unknown Journal';

export const getBestYear = (match) =>
  match?.metadata?.crossref_year ||
  match?.metadata?.publication_year ||
  null;

export const getBestAuthors = (match) =>
  Array.isArray(match?.metadata?.crossref_authors) && match.metadata.crossref_authors.length > 0
    ? match.metadata.crossref_authors
    : match?.metadata?.authors || [];

export const getCitationCount = (match) =>
  typeof match?.metadata?.citation_count === 'number'
    ? match.metadata.citation_count
    : 0;

export const hasRichMetadata = (match) =>
  !!(match?.metadata?.doi && match?.metadata?.crossref_journal);

export const countFrequency = (array) =>
  array.reduce((acc, item) => {
    if (!item) return acc;
    acc[item] = (acc[item] || 0) + 1;
    return acc;
  }, {}); 