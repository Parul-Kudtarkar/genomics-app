import React, { useState } from 'react';
import styled from 'styled-components';
import FilterPanel from './FilterPanel';
import FilterPills from './FilterPills';

const Card = styled.section`
  background: #fff;
  border: 1px solid #f5f5f7;
  border-radius: 24px;
  padding: 2.5rem 2rem 2rem 2rem;
  margin: 0 auto 2rem auto;
  box-shadow: 0 4px 24px rgba(0,0,0,0.04), 0 1px 4px rgba(0,0,0,0.02);
  max-width: 700px;
  width: 100%;
  position: relative;
`;
const Label = styled.label`
  font-size: 0.95rem;
  font-weight: 600;
  color: #1d1d1f;
  margin-bottom: 0.5rem;
  display: block;
`;
const TextArea = styled.textarea`
  width: 100%;
  min-height: 90px;
  padding: 1rem;
  border: 2px solid #e5e5e7;
  border-radius: 16px;
  background: #fff;
  color: #1d1d1f;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 1.2rem;
  transition: border-color 0.2s;
  &:focus {
    outline: none;
    border-color: #007AFF;
    box-shadow: 0 0 0 4px rgba(0,122,255,0.08);
  }
`;
const Select = styled.select`
  padding: 0.7rem 1.2rem;
  border: 2px solid #e5e5e7;
  border-radius: 16px;
  background: #fff;
  color: #1d1d1f;
  font-size: 1rem;
  font-family: inherit;
  margin-bottom: 1.2rem;
  transition: border-color 0.2s;
  &:focus {
    outline: none;
    border-color: #007AFF;
    box-shadow: 0 0 0 4px rgba(0,122,255,0.08);
  }
`;
const Button = styled.button`
  padding: 1rem 2.2rem;
  background: linear-gradient(135deg, #007AFF 0%, #5856D6 25%, #AF52DE 50%, #FF2D92 75%, #FF9500 100%);
  border: none;
  border-radius: 16px;
  color: white;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: box-shadow 0.2s, transform 0.1s;
  box-shadow: 0 4px 20px rgba(0,122,255,0.12);
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,122,255,0.18);
  }
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    background: #d1d1d6;
  }
`;
const FiltersBtn = styled.button`
  background: #f5f5f7;
  color: #007AFF;
  border: none;
  border-radius: 12px;
  padding: 0.6rem 1.2rem;
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 1.2rem;
  margin-right: 1rem;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
  &:hover {
    background: #e5e5e7;
  }
`;

const defaultFilters = {
  timePeriod: 'all',
  citationLevel: 'all',
  dataQuality: 'all',
};

export default function AdvancedSearchCard({ onSearch, loading }) {
  const [question, setQuestion] = useState('');
  const [model, setModel] = useState('gpt-4o');
  const [filters, setFilters] = useState(defaultFilters);
  const [showFilters, setShowFilters] = useState(false);

  const handleRemoveFilter = (category) => {
    setFilters(f => ({ ...f, [category]: defaultFilters[category] }));
  };

  const handleSubmit = e => {
    e.preventDefault();
    if (question.trim()) {
      onSearch({ query: question.trim(), model, filters });
    }
  };

  return (
    <Card>
      <FilterPills filters={filters} onRemove={handleRemoveFilter} />
      <FiltersBtn type="button" onClick={() => setShowFilters(v => !v)}>
        {showFilters ? 'Hide Filters' : 'Filters'}
      </FiltersBtn>
      <FilterPanel open={showFilters} onClose={() => setShowFilters(false)} filters={filters} onChange={setFilters} />
      <form onSubmit={handleSubmit}>
        <Label htmlFor="research-question">Research Question</Label>
        <TextArea
          id="research-question"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="Ask about diabetes-related research topic..."
          required
        />
        <Label htmlFor="model-select">AI Model</Label>
        <Select
          id="model-select"
          value={model}
          onChange={e => setModel(e.target.value)}
        >
                      <option value="gpt-4o">GPT-4o (Latest)</option>
            <option value="gpt-4o-mini">GPT-4o Mini (Fastest)</option>
            <option value="gpt-4-turbo">GPT-4 Turbo</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        </Select>
        <Button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Processing...' : 'Search & Analyze'}
        </Button>
      </form>
    </Card>
  );
} 