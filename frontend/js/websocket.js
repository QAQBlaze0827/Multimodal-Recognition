let ws = null;
let wsCallbacks = [];

function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws`;

    ws = new WebSocket(url);

    ws.onopen = () => {
        document.getElementById('ws-status').dataset.state = 'connected';
        document.getElementById('ws-status').textContent = typeof tr === 'function' ? tr('connected') : 'connected';
        document.getElementById('ws-status').style.color = 'var(--success)';
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            wsCallbacks.forEach(cb => cb(data));
        } catch (e) {
            console.warn('WS parse error:', e);
        }
    };

    ws.onclose = () => {
        document.getElementById('ws-status').dataset.state = 'disconnected';
        document.getElementById('ws-status').textContent = typeof tr === 'function' ? tr('disconnected') : 'disconnected';
        document.getElementById('ws-status').style.color = 'var(--danger)';
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        ws.close();
    };
}

function onWsMessage(callback) {
    wsCallbacks.push(callback);
}

function disconnectWebSocket() {
    if (ws) {
        ws.close();
        ws = null;
    }
}
