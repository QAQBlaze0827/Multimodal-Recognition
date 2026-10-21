const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function dashboard() {
    let now = Date.parse('2026-10-21T08:00:00Z');
    const elements = new Map();
    function element(key) {
        if (!elements.has(key)) {
            const classes = new Set();
            elements.set(key, {
                textContent: '', innerHTML: '', title: '', dataset: {}, style: {}, options: [],
                addEventListener() {},
                classList: {
                    toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
                    add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
                    contains(name) { return classes.has(name); },
                },
            });
        }
        return elements.get(key);
    }
    element('#ws-status').dataset.state = 'connected';
    const events = {}, intervals = [];
    const context = vm.createContext({
        console, Date: class extends Date { static now() { return now; } },
        window: { location: { protocol: 'http:', host: 'localhost' } },
        localStorage: { getItem: () => null, setItem() {} },
        document: {
            documentElement: {},
            getElementById: id => element(`#${id}`),
            querySelector: element, querySelectorAll: () => [],
            addEventListener: (name, fn) => { events[name] = fn; },
        },
        setInterval: fn => intervals.push(fn), setTimeout() {},
        WebSocket: class { close() { this.onclose(); } },
        getRecentLogs: () => Promise.resolve([]),
    });
    for (const file of ['charts.js', 'websocket.js', 'app.js']) {
        vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/js', file), 'utf8'), context);
    }
    const run = code => vm.runInContext(code, context);
    const send = (id = 1, age = 0, extra = {}) => {
        context.record = { id, timestamp: new Date(now - age).toISOString(),
            fused_emotion: 'happy', fused_conf: 0.8, video_emotion: 'happy', video_conf: 0.8,
            video_scores: { happy: 0.8, neutral: 0.2 }, face_detected: true, ...extra };
        run('acceptLiveData(record)');
    };
    return { run, send, element, context, events, intervals,
        advance(ms) { now += ms; run('refreshLiveFreshness()'); },
        state: () => run('liveFreshness().state'),
        count: () => run('liveTrend.length'),
    };
}

test('backend connection without records stays waiting', () => {
    const d = dashboard();
    d.run('refreshLiveFreshness()');
    assert.equal(d.state(), 'waiting');
    assert.equal(d.element('#live-data-label').textContent, '等待辨識資料');
});

test('fresh record expires at five seconds without a websocket message', () => {
    const d = dashboard(); d.send();
    assert.equal(d.state(), 'fresh');
    d.advance(4999); assert.equal(d.state(), 'fresh');
    d.advance(1); assert.equal(d.state(), 'stale');
    assert.equal(d.element('#ws-status').dataset.state, 'connected');
    assert.ok(d.element('#view-live').classList.contains('is-stale'));
    assert.equal(d.element('#view-live .live-summary h2').textContent, '最後一次判讀');
    assert.equal(d.element('header .dot').style.animationPlayState, 'paused');
});

test('an old initial record is stale immediately', () => {
    const d = dashboard(); d.send(1, 86400000);
    assert.equal(d.state(), 'stale');
    assert.match(d.element('#live-last-update').textContent, /86400/);
});

test('new data restores appearance and current-reading heading', () => {
    const d = dashboard(); d.send(); d.advance(6000); d.send(2);
    assert.equal(d.state(), 'fresh');
    assert.equal(d.element('#view-live').classList.contains('is-stale'), false);
    assert.equal(d.element('#view-live .live-summary h2').textContent, '目前判讀');
    assert.equal(d.element('header .dot').style.animationPlayState, 'running');
});

test('duplicate ID does not renew freshness or append trend', () => {
    const d = dashboard(); d.send(); d.advance(6000); d.send(1);
    assert.equal(d.state(), 'stale'); assert.equal(d.count(), 1);
});

test('reordered IDs do not replace the newest result', () => {
    const d = dashboard(); d.send(10); d.send(9, 10000, { fused_emotion: 'sad' });
    assert.equal(d.run('lastLiveData.id'), 10); assert.equal(d.count(), 1);
});

test('without IDs, duplicate and older timestamps are ignored', () => {
    const d = dashboard(); d.send(undefined, 0, { id: undefined });
    const stamp = d.run('lastLiveData.timestamp'); d.advance(6000);
    d.send(undefined, 0, { id: undefined, timestamp: stamp });
    d.send(undefined, 10000, { id: undefined });
    assert.equal(d.state(), 'stale'); assert.equal(d.count(), 1);
});

test('invalid, missing and future timestamps are not called live', () => {
    for (const timestamp of [undefined, '', 'bad-date', '2099-01-01T00:00:00Z']) {
        const d = dashboard(); d.send(1, 0, { timestamp });
        assert.equal(d.state(), 'unknown');
        assert.ok(d.element('#view-live').classList.contains('is-stale'));
    }
});

test('timezone-qualified timestamp is parsed as the same instant', () => {
    const d = dashboard(); d.send(1, 0, { timestamp: '2026-10-21T16:00:00+08:00' });
    assert.equal(d.state(), 'fresh');
});

test('language toggle updates stale labels without adding a reading or renewing age', () => {
    const d = dashboard(); d.send(); d.advance(6000);
    d.run("currentLang = 'en'; applyLanguage()");
    assert.equal(d.element('#live-data-label').textContent, 'Data updates stopped');
    assert.equal(d.element('#view-live .live-summary h2').textContent, 'Last Reading');
    assert.equal(d.state(), 'stale'); assert.equal(d.count(), 1);
    d.run("currentLang = 'zh'; applyLanguage()"); assert.equal(d.count(), 1);
});

test('websocket disconnect immediately marks results unverified, reconnect alone cannot revive stale data', () => {
    const d = dashboard(); d.run('connectWebSocket()'); d.send();
    d.run('ws.onclose()'); assert.equal(d.state(), 'disconnected');
    assert.equal(d.element('#live-data-label').textContent, '無法確認資料更新');
    d.advance(6000); d.run('ws.onopen()'); assert.equal(d.state(), 'stale');
    d.send(2); assert.equal(d.state(), 'fresh');
});

test('timer and background-tab updates continue even while another view is selected', () => {
    const d = dashboard();
    d.run('initHistoryView = initAnalyticsView = initReplayView = () => {}');
    d.events.DOMContentLoaded();
    assert.equal(d.intervals.length, 1);
    d.run("currentView = 'history'; ws.onopen()");
    d.send(1); d.advance(6000);
    d.context.next = { id: 2, timestamp: d.run('new Date(Date.now()).toISOString()'), fused_emotion: 'sad', fused_conf: 0.7 };
    d.run('ws.onmessage({data: JSON.stringify(next)})');
    assert.equal(d.run('lastLiveData.id'), 2);
    d.run("switchView('live')"); assert.equal(d.state(), 'fresh');
    d.advance(6000); d.events.visibilitychange(); assert.equal(d.state(), 'stale');
});

test('late initial HTTP response cannot overwrite a newer websocket record', async () => {
    const d = dashboard(); let resolve;
    d.context.getRecentLogs = () => new Promise(done => { resolve = done; });
    d.run('initLiveView()'); d.send(20);
    resolve([{ id: 19, timestamp: '2026-10-21T07:00:00Z', fused_emotion: 'sad' }]);
    await new Promise(done => setImmediate(done));
    assert.equal(d.run('lastLiveData.id'), 20); assert.equal(d.count(), 1);
});

test('invalid websocket payload does not replace the current record', () => {
    const d = dashboard(); d.send();
    d.run('acceptLiveData(null); acceptLiveData([]); acceptLiveData("bad")');
    assert.equal(d.run('lastLiveData.id'), 1);
});
