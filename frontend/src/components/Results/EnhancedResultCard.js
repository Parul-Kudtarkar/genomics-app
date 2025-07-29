import React from 'react';
import styled from 'styled-components';
import { getBestTitle, getBestJournal, getBestYear, getBestAuthors, getCitationCount, hasRichMetadata } from '../../utils/metadataHelpers';

const Card = styled.div`
  background: #fff;
  border: 1px solid #f5f5f7;
  border-radius: 20px;
  padding: 2rem 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 12px rgba(0,0,0,0.03);
  position: relative;
  overflow: hidden;
`;
const Title = styled.h3`
  font-size: 1.2rem;
  font-weight: 700;
  color: #1d1d1f;
  margin-bottom: 0.5rem;
`;
const MetaRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  margin-bottom: 0.7rem;
  align-items: center;
`;
const Badge = styled.span`
  background: #e5e5e7;
  color: #1d1d1f;
  border-radius: 12px;
  padding: 3px 12px;
  font-size: 0.85rem;
  font-weight: 500;
`;
const EnrichedBadge = styled(Badge)`
  background: #007AFF;
  color: #fff;
`;
const CitationBadge = styled(Badge)`
  background: linear-gradient(135deg, #30D158 0%, #00C896 100%);
  color: #fff;
`;
const Content = styled.p`
  color: #424242;
  line-height: 1.5;
  font-size: 0.97rem;
  margin-bottom: 0.7rem;
`;
const Authors = styled.div`
  font-size: 0.92rem;
  color: #6e6e73;
  margin-bottom: 0.2rem;
`;

export default function EnhancedResultCard({ match }) {
  // TEMPORARILY SIMPLIFIED FOR DEBUGGING
  console.log('EnhancedResultCard match:', match);
  
  const title = getBestTitle(match);
  const journal = getBestJournal(match);
  const year = getBestYear(match);
  const authors = getBestAuthors(match);
  const citations = getCitationCount(match);
  const hasRichData = hasRichMetadata(match);
  const chunkType = match?.metadata?.chunk_type || 'content';

  console.log('Authors:', authors, 'Type:', typeof authors);

  return (
    <Card>
      <Title>{title}</Title>
      <MetaRow>
        {journal && <Badge>{journal}</Badge>}
        {year && <Badge>{year}</Badge>}
        <Badge>{chunkType}</Badge>
        {citations > 0 && <CitationBadge>{citations} citations</CitationBadge>}
        {hasRichData && <EnrichedBadge>Enriched</EnrichedBadge>}
      </MetaRow>
      <Content>
        {typeof match.content === 'string' ? match.content.slice(0, 300) + (match.content.length > 300 ? '...' : '') : (match.content || '')}
      </Content>
      {Array.isArray(authors) && authors.length > 0 && (
        <Authors>
          Authors: {authors.slice(0,3).join(', ')}{match.metadata?.author_count > 3 ? `, +${match.metadata.author_count - 3} more` : ''}
        </Authors>
      )}
    </Card>
  );
} 