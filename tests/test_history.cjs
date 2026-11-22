const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness() {
    function element() {
        return { children: [], textContent: '', value: '30', hidden: false, options: [{}, {}, {}, {}],
            append(...children) { this.children.push(...children); },
            replaceChildren(...children) { this.children = children; },
            setAttribute() {}, addEventListener() {}, focus() {},
        };
    }
    const nodes = new Map();
    const get = id => { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); };
    const requests = [];
    const context = vm.createContext({ currentLang: 'zh', Date, console,
        formatEmotion: emotion => emotion, tr: key => key,
        document: { createElement: element, getElementById: get, querySelector: get },
        apiGet: (url, params) => new Promise((resolve, reject) => requests.push({ url, params, resolve, reject })),
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/js/history.js'), 'utf8'), context);
    return { get, requests, run: code => vm.runInContext(code, context) };
}
const item = { session_id: 's', first_record: '2026-11-21T10:00:00', last_record: '2026-11-21T10:01:00',
    total_logs: 4, valid_logs: 2, low_score_logs: 1, no_source_logs: 2, unresolved_logs: 0,
    distribution: { neutral: 0, happy: 1, sad: 1, anger: 0 } };

test('ties, zero-valid and English summaries do not invent a winner', () => {
    const h = harness();
    assert.match(h.run(`historyTop(${JSON.stringify(item)})`), /並列最多.*50%/);
    assert.match(h.run(`historyTop(${JSON.stringify({ ...item, valid_logs: 0, distribution: {} })})`), /沒有可參考/);
    h.run("currentLang = 'en'");
    assert.match(h.run(`historyTop(${JSON.stringify(item)})`), /Tied most frequent/);
});
test('summary loading clears old data; stale responses cannot replace latest', async () => {
    const h = harness();
    const first = h.run('loadHistory()');
    const second = h.run('loadHistory(10)');
    assert.equal(h.requests[1].params.offset, 10);
    h.requests[1].resolve({ total: 11, items: [item] });
    await second;
    h.requests[0].resolve({ total: 0, items: [] });
    await first;
    assert.equal(h.get('history-sessions').children.length, 1);
    const failed = h.run('loadHistory()');
    assert.equal(h.get('history-sessions').children.length, 0);
    h.requests[2].reject(new Error('offline'));
    await failed;
    assert.match(h.get('history-status').textContent, /載入失敗/);
});
test('excluded details hide fallback emotion and score, low scores remain visible', async () => {
    const h = harness();
    h.run(`historySelected = ${JSON.stringify(item)}`);
    const pending = h.run('loadHistoryDetail()');
    h.requests[0].resolve({ total: 2, items: [
        { timestamp: item.first_record, quality: 'no_source', fused_emotion: 'neutral', fused_conf: 1 },
        { timestamp: item.first_record, quality: 'valid', fused_emotion: 'sad', fused_conf: .4 },
    ] });
    await pending;
    const rows = h.get('#history-records tbody').children;
    assert.equal(rows[0].children[1].textContent, '--');
    assert.equal(rows[0].children[2].textContent, '--');
    assert.equal(rows[1].children[2].textContent, '40%');
    assert.match(rows[1].children[3].textContent, /低分數/);
});
test('changing period invalidates pending details and keeps them hidden', async () => {
    const h = harness();
    h.run(`historySelected = ${JSON.stringify(item)}`);
    const detail = h.run('loadHistoryDetail()');
    const summary = h.run('loadHistory()');
    h.requests[1].resolve({ total: 0, items: [] });
    await summary;
    h.requests[0].resolve({ total: 1, items: [{ quality: 'valid', fused_conf: 1 }] });
    await detail;
    assert.equal(h.get('history-detail').hidden, true);
    assert.equal(h.get('#history-records tbody').children.length, 0);
});
