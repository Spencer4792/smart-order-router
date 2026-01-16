const API_BASE = '/api/v1';

export async function fetchQuote(symbol) {
  const response = await fetch(`${API_BASE}/quote/${symbol}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch quote: ${response.statusText}`);
  }
  return response.json();
}

export async function routeOrder(orderRequest) {
  const response = await fetch(`${API_BASE}/route`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(orderRequest),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to route order');
  }
  return response.json();
}

export async function benchmarkAlgorithms(benchmarkRequest) {
  const response = await fetch(`${API_BASE}/benchmark`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(benchmarkRequest),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Benchmark failed');
  }
  return response.json();
}

export async function fetchAlgorithms() {
  const response = await fetch(`${API_BASE}/algorithms`);
  if (!response.ok) {
    throw new Error('Failed to fetch algorithms');
  }
  return response.json();
}

export async function fetchVenues(includeDarkPools = false) {
  const response = await fetch(`${API_BASE}/venues?include_dark_pools=${includeDarkPools}`);
  if (!response.ok) {
    throw new Error('Failed to fetch venues');
  }
  return response.json();
}
