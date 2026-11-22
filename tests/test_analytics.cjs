const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function dashboard() {
    const elements = new Map();
    const charts = [];
    const pending = [];
    function element(id) {
        if (!elements.has(id)) {
            let html = '', text = '';
            elements.set(id, {
                hidden: false, value: '30', attributes: {},
                setAttribute(k, v) { this.attributes[k] = v; },
                get textContent() { return text; },
                set textContent(value) { text = value; html = ''; },
                get innerHTML() { return html; },
                set innerHTML(value) { html = value; text = ''; },
                addEventListener() {},
            });
        }
        return elements.get(id);
    }
    const context = vm.createContext({
        console: { error() {} }, window: {},
        localStorage: { getItem: () => 'zh' },
        document: { addEventListener() {}, getElementById: element },
        Chart: class {
            constructor(_, config) { this.config = config; this.destroyed = false; charts.push(this); }
            destroy() { this.destroyed = true; }
        },
        getAnalytics: days => new Promise((resolve, reject) => pending.push({ days, resolve, reject })),
    });
    for (const file of ['charts.js', 'app.js']) {
        vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/js', file), 'utf8'), context);
    }
    return { element, charts, pending, context, run: code => vm.runInContext(code, context) };
}

const sample = {
    total_logs: 10, valid_logs: 6, excluded_logs: 4, no_source_logs: 3, unresolved_logs: 1,
    low_score_logs: 2, low_score_threshold: 0.5, session_count: 1,
    first_record: '2026-11-22T10:00:00', last_record: '2026-11-22T10:00:09',
    distribution: { happy: { count: 4, pct: 66.7, avg_conf: 0.7 }, neutral: { count: 2, pct: 33.3, avg_conf: 0.4 } },
};

async function load(d, data) {
    const promise = d.run('loadAnalytics()');
    d.pending.at(-1).resolve(data);
    await promise;
}

test('counts, denominator, low-score subset and source explanation are visible', async () => {
    const d = dashboard();
    await load(d, sample);
    assert.match(d.element('analytics-summary').innerHTML, /有效紀錄（分布分母）/);
    assert.match(d.element('analytics-summary').innerHTML, /其中低分數（低於 50%）/);
    assert.match(d.element('analytics-insight').textContent, /6 筆有效紀錄.*包含低分數/);
    assert.match(d.element('analytics-table-head').innerHTML, /平均模型分數/);
    assert.match(d.element('analytics-table-body').innerHTML, /66.7%/);
    assert.equal(d.charts.length, 1);
    assert.equal(d.element('analytics-distribution').hidden, false);
});

test('all-excluded and empty periods clear the previous chart and values', async () => {
    const d = dashboard();
    await load(d, sample);
    await load(d, { ...sample, valid_logs: 0, excluded_logs: 10, low_score_logs: 0, distribution: {} });
    assert.equal(d.charts[0].destroyed, true);
    assert.equal(d.element('analytics-distribution').hidden, true);
    assert.equal(d.element('analytics-table-body').innerHTML, '');
    assert.match(d.element('analytics-status').textContent, /有紀錄，但沒有/);
    await load(d, { ...sample, total_logs: 0, valid_logs: 0, excluded_logs: 0, no_source_logs: 0,
        unresolved_logs: 0, low_score_logs: 0, session_count: 0, distribution: {} });
    assert.match(d.element('analytics-status').textContent, /選取期間沒有紀錄/);
    assert.equal(d.element('analytics-range').textContent, '');
});

test('English labels and chart labels use current language', async () => {
    const d = dashboard();
    d.run("currentLang = 'en'; window.APP_LANG = 'en'");
    await load(d, sample);
    assert.match(d.element('analytics-table-head').innerHTML, /Mean model score/);
    assert.match(d.element('analytics-insight').textContent, /not accuracy/);
    assert.equal(d.charts[0].config.data.labels[0], 'Happy');
});

test('late responses cannot replace the selected period', async () => {
    const d = dashboard();
    const old = d.run('loadAnalytics()');
    d.element('analytics-days').value = '7';
    const recent = d.run('loadAnalytics()');
    assert.equal(d.pending[1].days, 7);
    d.pending[1].resolve(sample);
    await recent;
    d.pending[0].resolve({ ...sample, valid_logs: 0, distribution: {} });
    await old;
    assert.equal(d.element('analytics-distribution').hidden, false);
    assert.equal(d.charts.length, 1);
});

test('network errors and outdated backend remove stale results', async () => {
    const d = dashboard();
    await load(d, sample);
    const failed = d.run('loadAnalytics()');
    d.pending.at(-1).reject(new Error('offline'));
    await failed;
    assert.match(d.element('analytics-status').textContent, /載入失敗/);
    assert.equal(d.element('analytics-summary').innerHTML, '');
    await load(d, { total_logs: 10, distribution: sample.distribution });
    assert.match(d.element('analytics-status').textContent, /後端已更新/);
});

test('table remains usable without chart library and timestamps stay text', async () => {
    const d = dashboard();
    d.run('Chart = undefined');
    await load(d, { ...sample, first_record: '<img src=x onerror=alert(1)>' });
    assert.match(d.element('analytics-status').textContent, /圖表元件未載入/);
    assert.match(d.element('analytics-table-body').innerHTML, /66.7%/);
    assert.equal(d.element('analytics-range').innerHTML, '');
});
