const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function dashboard() {
    const nodes = new Map(), pending = [], timers = new Map(), events = {};
    let timerId = 0;
    function element() {
        return { textContent: '', value: '', style: {}, options: [], attributes: {}, listeners: {},
            setAttribute(k, v) { this.attributes[k] = v; }, removeAttribute(k) { delete this.attributes[k]; },
            addEventListener(k, fn) { this.listeners[k] = fn; },
            replaceChildren() { this.options = []; }, append(option) { this.options.push(option); },
        };
    }
    const get = id => { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); };
    const context = vm.createContext({ console, Date, window: {},
        localStorage: { getItem: () => 'zh' },
        document: { getElementById: get, createElement: element, hidden: false,
            addEventListener: (name, fn) => { events[name] = fn; } },
        setInterval(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms }); return id; },
        clearInterval(id) { timers.delete(id); },
        getSessionTimeline: id => new Promise((resolve, reject) => pending.push({ id, resolve, reject })),
        getSessions: () => Promise.resolve([]),
    });
    for (const file of ['charts.js', 'app.js']) vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/js', file), 'utf8'), context);
    const run = code => vm.runInContext(code, context);
    get('replay-session').value = 'sample';
    run('initReplayView()');
    return { get, run, pending, timers, events, context };
}
const rows = [
    { timestamp: '2026-11-22T10:00:00', quality: 'valid', fused_emotion: 'happy', fused_conf: .8, video_emotion: 'happy', video_conf: .8 },
    { timestamp: '2026-11-22T10:00:01', quality: 'no_source', fused_emotion: null, fused_conf: null },
    { timestamp: '2026-11-22T10:00:20', quality: 'valid', fused_emotion: 'sad', fused_conf: .4, audio_emotion: 'sad', audio_conf: .4 },
];
async function load(h, data = rows) {
    const promise = h.run('startReplay()');
    h.pending.at(-1).resolve(data);
    await promise;
}
test('invalid record clears previous emotion and both source outputs', async () => {
    const h = dashboard(); await load(h);
    assert.equal(h.get('replay-emotion').textContent, '開心');
    h.run('seekReplay(1)');
    assert.equal(h.get('replay-emotion').textContent, '--');
    assert.equal(h.get('replay-conf').textContent, '--');
    assert.equal(h.get('replay-video').textContent, '無有效來源');
    assert.match(h.get('replay-insight').textContent, /無有效訊號/);
    h.run('seekReplay(2)');
    assert.match(h.get('replay-audio').textContent, /悲傷.*40%/);
    assert.match(h.get('replay-insight').textContent, /分數偏低/);
    assert.match(h.get('replay-gap').textContent, /19.0 秒/);
});
test('playback stops at end, restart begins at first record, speed takes effect immediately', async () => {
    const h = dashboard(); await load(h);
    h.run('toggleReplayPlay()');
    assert.equal([...h.timers.values()][0].ms, 1000);
    h.get('replay-speed').listeners.change({ target: { value: '5' } });
    assert.equal(h.timers.size, 1);
    assert.equal([...h.timers.values()][0].ms, 200);
    [...h.timers.values()][0].fn(); [...h.timers.values()][0].fn();
    assert.equal(h.timers.size, 0);
    assert.match(h.get('replay-status').textContent, /最後/);
    h.run('toggleReplayPlay()');
    assert.equal(h.run('replayIndex'), 0);
    h.get('replay-timeline').listeners.input({ target: { value: '1' } });
    assert.equal(h.timers.size, 0);
    assert.equal(h.run('replayIndex'), 1);
});
test('loading clears previous results and outdated response cannot replace newer session', async () => {
    const h = dashboard(); await load(h);
    const old = h.run('startReplay()');
    assert.equal(h.get('replay-emotion').textContent, '--');
    h.get('replay-session').value = 'new';
    const recent = h.run('startReplay()');
    h.pending[2].resolve([]); await recent;
    h.pending[1].resolve(rows); await old;
    assert.equal(h.run('replayData.length'), 0);
    assert.equal(h.get('replay-play').disabled, true);
    assert.match(h.get('replay-status').textContent, /沒有可回放/);
});
test('error and old backend clear state; singleton disables playback', async () => {
    const h = dashboard(); await load(h, [rows[0]]);
    assert.equal(h.get('replay-play').disabled, true);
    await load(h, [{ fused_emotion: 'happy' }]);
    assert.match(h.get('replay-status').textContent, /載入失敗/);
    const pending = h.run('startReplay()');
    h.pending.at(-1).reject(new Error('offline')); await pending;
    assert.equal(h.get('replay-summary').textContent, '');
});
test('language refresh preserves position and document hiding pauses playback', async () => {
    const h = dashboard(); await load(h);
    h.run("seekReplay(2); currentLang = 'en'; window.APP_LANG = 'en'; updateReplayStaticLabels(); seekReplay(replayIndex)");
    assert.match(h.get('replay-status').textContent, /End of records/);
    assert.match(h.get('replay-conf').textContent, /Model score/);
    assert.equal(h.run('replayIndex'), 2);
    h.run('toggleReplayPlay()');
    h.context.document.hidden = true; h.events.visibilitychange();
    assert.equal(h.timers.size, 0);
});

test('all-invalid sessions stay replayable without displaying fallback emotions', async () => {
    const h = dashboard(); await load(h, [rows[1], { ...rows[1], quality: 'unresolved' }]);
    assert.match(h.get('replay-summary').textContent, /可參考 0/);
    assert.equal(h.get('replay-play').disabled, false);
    h.run('seekReplay(1)');
    assert.equal(h.get('replay-emotion').textContent, '--');
    assert.match(h.get('replay-insight').textContent, /綜合結果無效/);
});

test('session list failure can retry and safely renders labels without raw HTML', async () => {
    const h = dashboard();
    h.context.getSessions = () => Promise.reject(new Error('offline'));
    await h.run('loadReplaySessions()');
    assert.match(h.get('replay-status').textContent, /載入失敗/);
    h.context.getSessions = () => Promise.resolve([{ id: '<img onerror=alert(1)>', start_time: '<script>', log_count: 2 }]);
    await h.get('replay-refresh').listeners.click();
    assert.equal(h.get('replay-load').disabled, false);
    assert.equal(h.get('replay-session').options[1].value, '<img onerror=alert(1)>');
    assert.equal(h.get('replay-session').options[1].textContent, '-- · 2 筆');
});
