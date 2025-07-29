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

export const getBestAuthors = (match) => {
  const crossrefAuthors = match?.metadata?.crossref_authors;
  const authors = match?.metadata?.authors;
  
  if (Array.isArray(crossrefAuthors) && crossrefAuthors.length > 0) {
    return crossrefAuthors;
  }
  
  if (Array.isArray(authors)) {
    return authors;
  }
  
  // If authors is a string, split it
  if (typeof authors === 'string') {
    return authors.split(',').map(author => author.trim()).filter(author => author.length > 0);
  }
  
  return [];
};

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