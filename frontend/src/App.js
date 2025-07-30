import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Auth0Provider } from '@auth0/auth0-react';
import styled, { createGlobalStyle } from 'styled-components';
import { auth0Config } from './auth/auth0-config';
import { getBestTitle } from './utils/metadataHelpers';
import AdvancedSearchCard from './components/Search/AdvancedSearchCard';
import EnhancedResultCard from './components/Results/EnhancedResultCard';
import VectorStoreContents from './components/VectorStore/VectorStoreContents';
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
  justify-content: space-between;
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

const CollapsibleSection = styled.div`
  margin-bottom: 2rem;
`;

const SectionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
`;

const ToggleSwitch = styled.label`
  display: flex;
  align-items: center;
  cursor: pointer;
  font-size: 0.9rem;
  color: #6e6e73;
`;

const Switch = styled.input`
  margin-left: 0.5rem;
  margin-right: 0.5rem;
`;

const CollapseButton = styled.button`
  background: none;
  border: none;
  color: #007AFF;
  font-size: 1.2rem;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 8px;
  transition: background 0.2s;
  &:hover {
    background: #f0f0f0;
  }
`;

const TabContainer = styled.div`
  margin-bottom: 1rem;
`;

const TabButtons = styled.div`
  display: flex;
  border-bottom: 2px solid #e5e5e7;
  margin-bottom: 1rem;
`;

const TabButton = styled.button`
  background: none;
  border: none;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: ${props => props.active ? '#007AFF' : '#6e6e73'};
  border-bottom: 2px solid ${props => props.active ? '#007AFF' : 'transparent'};
  cursor: pointer;
  transition: color 0.2s;
  &:hover {
    color: #007AFF;
  }
`;

const MainTabContainer = styled.div`
  margin-bottom: 2rem;
`;

const MainTabButtons = styled.div`
  display: flex;
  border-bottom: 2px solid #e5e5e7;
  margin-bottom: 1.5rem;
`;

const MainTabButton = styled.button`
  background: none;
  border: none;
  padding: 1rem 2rem;
  font-size: 1.1rem;
  font-weight: 600;
  color: ${props => props.active ? '#007AFF' : '#6e6e73'};
  border-bottom: 3px solid ${props => props.active ? '#007AFF' : 'transparent'};
  cursor: pointer;
  transition: color 0.2s;
  &:hover {
    color: #007AFF;
  }
`;

const MainTabContent = styled.div`
  display: ${props => props.active ? 'block' : 'none'};
`;

const TabContent = styled.div`
  display: ${props => props.active ? 'block' : 'none'};
`;

const AccordionItem = styled.div`
  border: 1px solid #e5e5e7;
  border-radius: 12px;
  margin-bottom: 0.5rem;
  overflow: hidden;
`;

const AccordionHeader = styled.button`
  width: 100%;
  background: #f5f5f7;
  border: none;
  padding: 1rem 1.5rem;
  text-align: left;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: background 0.2s;
  &:hover {
    background: #e5e5e7;
  }
`;

const AccordionContent = styled.div`
  padding: 1rem 1.5rem;
  background: #fff;
  border-top: 1px solid #e5e5e7;
  display: ${props => props.expanded ? 'block' : 'none'};
`;

const Footer = styled.footer`
  text-align: center;
  padding: 2rem 0;
  color: #86868b;
  font-size: 0.9rem;
  border-top: 1px solid #e5e5e7;
  margin-top: auto;
`;

function ResearchApp() {
  const [results, setResults] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  const [showReasoning, setShowReasoning] = useState(true);
  const [activeTab, setActiveTab] = useState('answer');
  const [expandedSteps, setExpandedSteps] = useState({});
  const [mainTab, setMainTab] = useState('search');
  const apiClient = useApiClient();

  // Parse CoT response into steps
  const parseCoTResponse = (response) => {
    if (!response) return { steps: [], finalAnswer: '' };
    
    const lines = response.split('\n');
    const steps = [];
    let finalAnswer = '';
    let currentStep = null;
    
    for (const line of lines) {
      if (line.startsWith('Step ')) {
        if (currentStep) steps.push(currentStep);
        const stepMatch = line.match(/Step (\d+): (.+)/);
        if (stepMatch) {
          currentStep = {
            number: parseInt(stepMatch[1]),
            title: stepMatch[2].replace(/[\[\]]/g, ''),
            content: ''
          };
        }
      } else if (line.startsWith('Final Answer:')) {
        if (currentStep) steps.push(currentStep);
        finalAnswer = line.replace('Final Answer:', '').trim();
        break;
      } else if (currentStep && line.trim()) {
        currentStep.content += line.trim() + ' ';
      }
    }
    
    if (currentStep) steps.push(currentStep);
    
    return { steps, finalAnswer };
  };

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
      <div style={{width: '100%'}}>
        <Header>
          <HeaderLeft>
            <Title>RAG-Enhanced Machine Learning for Diabetes Literature</Title>
            <Subtitle></Subtitle>
          </HeaderLeft>
        </Header>
        <MainContent>
          <MainTabContainer>
            <MainTabButtons>
              <MainTabButton 
                active={mainTab === 'search'} 
                onClick={() => setMainTab('search')}
              >
                 Search & Analyze
              </MainTabButton>
              <MainTabButton 
                active={mainTab === 'contents'} 
                onClick={() => setMainTab('contents')}
              >
                KOI's library
              </MainTabButton>
            </MainTabButtons>
            
            <MainTabContent active={mainTab === 'search'}>
              <AdvancedSearchCard onSearch={handleSearch} loading={loading} />
              {loading && <Loading>KOI is thinking...</Loading>}
              {error && <ErrorMsg>{error}</ErrorMsg>}

              {results && (
                <>
                  {results.llm_response && (
                    <CollapsibleSection>
                      <SectionHeader>
                        <SectionTitle>KOI's Analysis</SectionTitle>
                        <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                          <ToggleSwitch>
                            Show reasoning
                            <Switch 
                              type="checkbox" 
                              checked={showReasoning}
                              onChange={(e) => setShowReasoning(e.target.checked)}
                            />
                          </ToggleSwitch>
                          <CollapseButton onClick={() => setShowReasoning(!showReasoning)}>
                            {showReasoning ? '−' : '+'}
                          </CollapseButton>
                        </div>
                      </SectionHeader>
                      
                      {showReasoning && (
                        <TabContainer>
                          <TabButtons>
                            <TabButton 
                              active={activeTab === 'answer'} 
                              onClick={() => setActiveTab('answer')}
                            >
                              Final Answer
                            </TabButton>
                            <TabButton 
                              active={activeTab === 'reasoning'} 
                              onClick={() => setActiveTab('reasoning')}
                            >
                              Reasoning Steps
                            </TabButton>
                          </TabButtons>
                          
                          <TabContent active={activeTab === 'answer'}>
                            <div style={{background: '#f5f5f7', borderRadius: 12, padding: '1rem 1.5rem', color: '#1d1d1f', lineHeight: '1.6'}}>
                              {parseCoTResponse(results.llm_response).finalAnswer || results.llm_response}
                            </div>
                          </TabContent>
                          
                          <TabContent active={activeTab === 'reasoning'}>
                            <div>
                              {parseCoTResponse(results.llm_response).steps.map((step, index) => (
                                <AccordionItem key={index}>
                                  <AccordionHeader 
                                    onClick={() => setExpandedSteps(prev => ({
                                      ...prev,
                                      [index]: !prev[index]
                                    }))}
                                  >
                                    <span>Step {step.number}: {step.title}</span>
                                    <span>{expandedSteps[index] ? '−' : '+'}</span>
                                  </AccordionHeader>
                                  <AccordionContent expanded={expandedSteps[index]}>
                                    {step.content || 'No detailed content available for this step.'}
                                  </AccordionContent>
                                </AccordionItem>
                              ))}
                            </div>
                          </TabContent>
                        </TabContainer>
                      )}
                    </CollapsibleSection>
                  )}
                  
                  <section>
                    <h2 style={{fontSize: '1.2rem', fontWeight: 600, margin: '1.5rem 0 1rem 0'}}>
                      Source Publications ({results.matches?.length || 0})
                    </h2>
                    <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem'}}>
                      {results.matches?.map((match, idx) => (
                        <EnhancedResultCard key={match.id || idx} match={match} />
                      ))}
                    </div>
                  </section>
                </>
              )}
            </MainTabContent>
            
            <MainTabContent active={mainTab === 'contents'}>
              <VectorStoreContents />
            </MainTabContent>
          </MainTabContainer>
        </MainContent>
      </div>
      <Footer>
        © GaultonLab 2025. All rights reserved.
      </Footer>
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
