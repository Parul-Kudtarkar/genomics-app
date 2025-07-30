import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useApiClient } from '../../utils/apiClient';

const Container = styled.div`
  background: #fff;
  border-radius: 16px;
  border: 1px solid #e5e5e7;
  overflow: hidden;
`;

const Header = styled.div`
  background: linear-gradient(135deg, #007AFF 0%, #5856D6 25%, #AF52DE 50%, #FF2D92 75%, #FF9500 100%);
  color: white;
  padding: 1.5rem;
  text-align: center;
`;

const Title = styled.h2`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
`;

const Subtitle = styled.p`
  margin: 0.5rem 0 0 0;
  opacity: 0.9;
  font-size: 0.9rem;
`;

const StatsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  padding: 1.5rem;
  background: #f5f5f7;
  border-bottom: 1px solid #e5e5e7;
`;

const StatCard = styled.div`
  background: white;
  padding: 1rem;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
`;

const StatNumber = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  color: #007AFF;
  margin-bottom: 0.25rem;
`;

const StatLabel = styled.div`
  font-size: 0.8rem;
  color: #6e6e73;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const PapersContainer = styled.div`
  max-height: 400px;
  overflow-y: auto;
  padding: 1rem;
`;

const PaperItem = styled.div`
  padding: 1rem;
  border: 1px solid #e5e5e7;
  border-radius: 12px;
  margin-bottom: 0.75rem;
  background: #fff;
  transition: box-shadow 0.2s;
  
  &:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }
`;

const PaperTitle = styled.div`
  font-weight: 600;
  color: #1d1d1f;
  margin-bottom: 0.5rem;
  line-height: 1.4;
`;

const PaperMeta = styled.div`
  display: flex;
  gap: 1rem;
  font-size: 0.8rem;
  color: #6e6e73;
  flex-wrap: wrap;
`;

const MetaItem = styled.span`
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const Badge = styled.span`
  background: ${props => props.source === 'pmc' ? '#007AFF' : '#34C759'};
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 600;
`;

const Loading = styled.div`
  text-align: center;
  padding: 2rem;
  color: #86868b;
`;

const Error = styled.div`
  color: #c53030;
  background: #fff5f5;
  border: 1px solid #fed7d7;
  border-radius: 12px;
  padding: 1rem;
  margin: 1rem;
`;

export default function VectorStoreContents({ preloadedContents, isLoading }) {
  const [contents, setContents] = useState(preloadedContents);
  const [loading, setLoading] = useState(isLoading);
  const [error, setError] = useState('');
  const apiClient = useApiClient();

  // Update state when props change
  useEffect(() => {
    setContents(preloadedContents);
    setLoading(isLoading);
  }, [preloadedContents, isLoading]);

  // Fallback to fetch if preloaded data is not available
  useEffect(() => {
    if (!preloadedContents && !loading) {
      const fetchContents = async () => {
        try {
          setLoading(true);
          const data = await apiClient.get('/vector-store/contents');
          setContents(data);
        } catch (err) {
          setError(err.message || 'Failed to load vector store contents');
        } finally {
          setLoading(false);
        }
      };

      fetchContents();
    }
  }, [preloadedContents, loading, apiClient]);

  if (loading) {
    return (
      <Container>
        <Header>
          <Title>Vector Store Contents</Title>
          <Subtitle>Papers available for search</Subtitle>
        </Header>
        <Loading>Loading contents...</Loading>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <Header>
          <Title>Vector Store Contents</Title>
          <Subtitle>Papers available for search</Subtitle>
        </Header>
        <Error>{error}</Error>
      </Container>
    );
  }

  if (!contents) {
    return (
      <Container>
        <Header>
          <Title>Vector Store Contents</Title>
          <Subtitle>Papers available for search</Subtitle>
        </Header>
        <Error>No data available</Error>
      </Container>
    );
  }

  const { papers, statistics } = contents;

  return (
    <Container>
      <Header>
        <Title>Vector Store Contents</Title>
        <Subtitle>Papers available for search</Subtitle>
      </Header>
      
      <StatsContainer>
        <StatCard>
          <StatNumber>{statistics.total_papers}</StatNumber>
          <StatLabel>Total Papers</StatLabel>
        </StatCard>
        <StatCard>
          <StatNumber>{statistics.total_chunks}</StatNumber>
          <StatLabel>Total Chunks</StatLabel>
        </StatCard>
        <StatCard>
          <StatNumber>{statistics.text_papers}</StatNumber>
          <StatLabel>Text Documents</StatLabel>
        </StatCard>
        <StatCard>
          <StatNumber>{statistics.pmc_papers}</StatNumber>
          <StatLabel>PMC Articles</StatLabel>
        </StatCard>
      </StatsContainer>
      
      <PapersContainer>
        {papers.map((paper) => (
          <PaperItem key={paper.id}>
            <PaperTitle>{paper.title}</PaperTitle>
            <PaperMeta>
              <MetaItem>
                <Badge source={paper.source}>
                  {paper.source === 'pmc' ? 'PMC' : 'TEXT'}
                </Badge>
              </MetaItem>
              <MetaItem>
                📄 {paper.chunk_count} chunks
              </MetaItem>
              <MetaItem>
                📅 {new Date(paper.processed_at).toLocaleDateString()}
              </MetaItem>
              <MetaItem>
                🏷️ {paper.type}
              </MetaItem>
            </PaperMeta>
          </PaperItem>
        ))}
      </PapersContainer>
    </Container>
  );
} 