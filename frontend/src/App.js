import React, { useState } from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import { getBestTitle } from './utils/metadataHelpers';
import AdvancedSearchCard from './components/Search/AdvancedSearchCard';
import EnhancedResultCard from './components/Results/EnhancedResultCard';

const GlobalStyle = createGlobalStyle`
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', 'Segoe UI', Roboto, sans-serif;
    background: #fff;
    color: #1d1d1f;
    margin: 0;
    min-height: 100vh;
  }
`;

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #fff;
`;

const Header = styled.header`
  text-align: center;
  margin: 2rem 0 1.5rem 0;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #007AFF 0%, #5856D6 25%, #AF52DE 50%, #FF2D92 75%, #FF9500 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
`;

const Subtitle = styled.p`
  color: #6e6e73;
  font-size: 1.2rem;
  margin-bottom: 0.5rem;
`;

const MainContent = styled.main`
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  padding: 0 1rem 2rem 1rem;
`;

const Loading = styled.div`
  margin: 2rem auto;
  font-size: 1.2rem;
  color: #86868b;
`;

const ErrorMsg = styled.div`
  color: #c53030;
  background: #fff5f5;
  border: 1px solid #fed7d7;
  border-radius: 12px;
  padding: 1rem 1.5rem;
  margin: 1rem 0;
`;

export default function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Main API call
  const handleSearch = async ({ query, model, filters }) => {
    setLoading(true);
    setError('');
    setResults(null);
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model, top_k: 5, temperature: 0.1, filters })
      });
      if (!response.ok) throw new Error('API error: ' + response.status);
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <GlobalStyle />
      <AppContainer>
        <Header>
          <Title>Diabetes Research Assistant</Title>
          <Subtitle>Explore the latest diabetes research with AI-powered search and analysis</Subtitle>
        </Header>
        <MainContent>
          <AdvancedSearchCard onSearch={handleSearch} loading={loading} />
          {loading && <Loading>Loading results...</Loading>}
          {error && <ErrorMsg>{error}</ErrorMsg>}

          {results && (
            <>
              {results.llm_response && (
                <section style={{marginBottom: '2rem'}}>
                  <h2 style={{fontSize: '1.5rem', fontWeight: 700, marginBottom: 8}}>AI Analysis</h2>
                  <div style={{background: '#f5f5f7', borderRadius: 12, padding: '1rem 1.5rem', color: '#1d1d1f'}}>
                    {results.llm_response.split('\n').map((p, i) => <p key={i}>{p}</p>)}
                  </div>
                </section>
              )}
              <section>
                <h2 style={{fontSize: '1.2rem', fontWeight: 600, margin: '1.5rem 0 1rem 0'}}>Source Publications ({results.matches?.length || 0})</h2>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem'}}>
                  {results.matches?.map((match, idx) => (
                    <EnhancedResultCard key={match.id || idx} match={match} />
                  ))}
                </div>
              </section>
            </>
          )}
        </MainContent>
      </AppContainer>
    </>
  );
}
