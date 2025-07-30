import { useApiClient } from './apiClient';
import { useState, useEffect } from 'react';

export const useDataPreloader = () => {
  const [preloadedData, setPreloadedData] = useState({
    vectorStoreContents: null,
    availableModels: null,
    filterOptions: null,
    status: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const apiClient = useApiClient();

  useEffect(() => {
    const preloadAllData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Preload all data in parallel
        const [
          vectorStoreContents,
          availableModels,
          filterOptions,
          status
        ] = await Promise.allSettled([
          apiClient.get('/vector-store/contents'),
          apiClient.get('/models'),
          apiClient.get('/filters/options'),
          apiClient.get('/status')
        ]);

        // Process results
        const newData = {
          vectorStoreContents: vectorStoreContents.status === 'fulfilled' ? vectorStoreContents.value : null,
          availableModels: availableModels.status === 'fulfilled' ? availableModels.value : null,
          filterOptions: filterOptions.status === 'fulfilled' ? filterOptions.value : null,
          status: status.status === 'fulfilled' ? status.value : null
        };

        setPreloadedData(newData);

        // Log any failed requests
        const failedRequests = [
          { name: 'Vector Store Contents', result: vectorStoreContents },
          { name: 'Available Models', result: availableModels },
          { name: 'Filter Options', result: filterOptions },
          { name: 'Status', result: status }
        ].filter(req => req.result.status === 'rejected');

        if (failedRequests.length > 0) {
          console.warn('Some data failed to preload:', failedRequests.map(req => req.name));
        }

      } catch (err) {
        console.error('Data preloading failed:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    preloadAllData();
  }, [apiClient]);

  return {
    preloadedData,
    loading,
    error,
    // Helper functions to get specific data
    getVectorStoreContents: () => preloadedData.vectorStoreContents,
    getAvailableModels: () => preloadedData.availableModels,
    getFilterOptions: () => preloadedData.filterOptions,
    getStatus: () => preloadedData.status
  };
}; 