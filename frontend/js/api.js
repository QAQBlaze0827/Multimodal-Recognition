const API_BASE = '/api';

async function apiGet(path, params = {}) {
    const url = new URL(path, window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
        if (v !== null && v !== undefined && v !== '') url.searchParams.set(k, v);
    });
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
    return res.json();
}

async function apiPost(path, body = {}) {
    const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
    return res.json();
}

async function getRecentLogs(n = 100) {
    return apiGet(`${API_BASE}/emotions/recent`, { n });
}

async function getLogs(params = {}) {
    return apiGet(`${API_BASE}/emotions`, params);
}

async function getSessions() {
    return apiGet(`${API_BASE}/sessions`);
}

async function getSessionDetail(id) {
    return apiGet(`${API_BASE}/sessions/${id}`);
}

async function getSessionLogs(id) {
    return apiGet(`${API_BASE}/sessions/${id}/logs`);
}

async function getSessionTimeline(id) {
    return apiGet(`${API_BASE}/sessions/${id}/timeline`);
}

async function getAnalytics(days = 30) {
    return apiGet(`${API_BASE}/analytics`, { days });
}
