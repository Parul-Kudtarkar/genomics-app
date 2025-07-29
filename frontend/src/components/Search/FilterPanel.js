import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

// Filter config - simplified for current functionality
const filterConfig = {
  timePeriods: [
    { value: 'recent', label: 'Last 2 Years', years: [2023, 2024] },
    { value: '5year', label: 'Last 5 Years', years: [2020, 2024] },
    { value: 'decade', label: 'Last 10 Years', years: [2015, 2024] },
    { value: 'custom', label: 'Custom Range' }
  ],
  citationLevels: [
    { value: 'high', label: 'High Impact (50+ citations)', min: 50 },
    { value: 'medium', label: 'Medium Impact (10-49)', min: 10, max: 49 },
    { value: 'emerging', label: 'Emerging (1-9)', min: 1, max: 9 },
    { value: 'all', label: 'All Papers' }
  ]
};

const PanelContainer = styled(motion.div)`
  background: #fff;
  border: 1px solid #e5e5e7;
  border-radius: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  padding: 2rem 1.5rem 1.5rem 1.5rem;
  margin-bottom: 1.5rem;
  max-width: 700px;
  width: 100%;
`;
const Section = styled.div`
  margin-bottom: 1.2rem;
`;
const SectionTitle = styled.div`
  font-size: 1.05rem;
  font-weight: 600;
  color: #1d1d1f;
  margin-bottom: 0.5rem;
`;
const OptionRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
`;
const OptionButton = styled.button`
  background: ${({ selected }) => selected ? 'linear-gradient(135deg, #007AFF 0%, #AF52DE 100%)' : '#f5f5f7'};
  color: ${({ selected }) => selected ? '#fff' : '#1d1d1f'};
  border: none;
  border-radius: 12px;
  padding: 0.5rem 1.1rem;
  font-size: 0.97rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
  &:hover {
    background: ${({ selected }) => selected ? 'linear-gradient(135deg, #007AFF 0%, #AF52DE 100%)' : '#e5e5e7'};
  }
`;
const PanelHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.2rem;
`;
const CloseBtn = styled.button`
  background: none;
  border: none;
  color: #86868b;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0 0.5rem;
`;

export default function FilterPanel({ open, onClose, filters, onChange }) {
  // Helper to update a filter
  const setFilter = (category, value) => {
    onChange({ ...filters, [category]: value });
  };

  return (
    <AnimatePresence>
      {open && (
        <PanelContainer
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          <PanelHeader>
            <span style={{fontWeight: 700, fontSize: '1.15rem'}}>Filters</span>
            <CloseBtn onClick={onClose} title="Close">×</CloseBtn>
          </PanelHeader>
          <div style={{fontSize: '0.9rem', color: '#86868b', marginBottom: '1rem', fontStyle: 'italic'}}>
            Note: Filters are disabled until metadata is fully implemented
          </div>
          <Section>
            <SectionTitle>Time Period</SectionTitle>
            <OptionRow>
              {filterConfig.timePeriods.map(opt => (
                <OptionButton
                  key={opt.value}
                  selected={filters.timePeriod === opt.value}
                  onClick={() => setFilter('timePeriod', opt.value)}
                  type="button"
                  disabled={true}
                  style={{ opacity: 0.5, cursor: 'not-allowed' }}
                >
                  {opt.label}
                </OptionButton>
              ))}
            </OptionRow>
          </Section>
          <Section>
            <SectionTitle>Citation Level</SectionTitle>
            <OptionRow>
              {filterConfig.citationLevels.map(opt => (
                <OptionButton
                  key={opt.value}
                  selected={filters.citationLevel === opt.value}
                  onClick={() => setFilter('citationLevel', opt.value)}
                  type="button"
                  disabled={true}
                  style={{ opacity: 0.5, cursor: 'not-allowed' }}
                >
                  {opt.label}
                </OptionButton>
              ))}
            </OptionRow>
          </Section>
        </PanelContainer>
      )}
    </AnimatePresence>
  );
} 