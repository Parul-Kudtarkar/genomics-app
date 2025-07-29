import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Auth0Provider } from '@auth0/auth0-react';
import styled, { createGlobalStyle } from 'styled-components';
import { auth0Config } from './auth/auth0-config';
import { getBestTitle } from './utils/metadataHelpers';
import AdvancedSearchCard from './components/Search/AdvancedSearchCard';
import EnhancedResultCard from './components/Results/EnhancedResultCard';
import { useApiClient } from './utils/apiClient';

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
  margin: 3rem 0 2rem 0;
  width: 100%;
  max-width: 1200px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0 2rem;
`;

const HeaderLeft = styled.div`
  text-align: left;
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
  max-width: 800px;
  margin: 0 auto;
  padding: 0 2rem 3rem 2rem;
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

function ResearchApp() {
  const [results, setResults] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  const apiClient = useApiClient();

  // Main API call with authentication
  const handleSearch = async ({ query, model, filters }) => {
    setLoading(true);
    setError('');
    setResults(null);
    try {
      const data = await apiClient.post('/query', { 
        query, 
        model, 
        top_k: 5, 
        temperature: 0.1, 
        filters 
      });
      setResults(data);
    } catch (err) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppContainer>
      <Header>
        <HeaderLeft>
          <Title>RAG-Enhanced Machine Learning for Diabetes Literature</Title>
          <Subtitle></Subtitle>
        </HeaderLeft>
      </Header>
      <MainContent>
        <AdvancedSearchCard onSearch={handleSearch} loading={loading} />
        {loading && <Loading>KOI is thinking...</Loading>}
        {error && <ErrorMsg>{error}</ErrorMsg>}

        {results && (
          <>
            {results.llm_response && (
              <section style={{marginBottom: '2rem'}}>
                <h2 style={{fontSize: '1.5rem', fontWeight: 700, marginBottom: 8}}>AI Analysis (Chain of Thought)</h2>
                <div style={{background: '#f5f5f7', borderRadius: 12, padding: '1rem 1.5rem', color: '#1d1d1f', whiteSpace: 'pre-wrap', lineHeight: '1.6'}}>
                  {results.llm_response}
                </div>
              </section>
            )}
            <section>
              <h2 style={{fontSize: '1.2rem', fontWeight: 600, margin: '1.5rem 0 1rem 0'}}>
                Source Publications ({results.matches?.length || 0})
                <span style={{fontSize: '0.9rem', fontWeight: 400, color: '#6e6e73', marginLeft: '0.5rem'}}>
                  (Unique papers)
                </span>
              </h2>
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
  );
}

export default function App() {
  return (
    // TEMPORARILY DISABLED AUTH0 - FOR TESTING ONLY
    // TODO: Re-enable Auth0Provider after testing
    <Router>
      <GlobalStyle />
      <Routes>
        <Route 
          path="/" 
          element={<ResearchApp />} 
        />
      </Routes>
    </Router>
  );
}
