const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function page() {
    const elements = new Map(), timers = [];
    const element = id => {
        if (!elements.has(id)) elements.set(id, { textContent: '', innerHTML: '', dataset: {} });
        return elements.get(id);
    };
    const sockets = [];
    const context = vm.createContext({
        Date, JSON, console, currentLang: 'zh',
        document: { getElementById: element, addEventListener() {} },
        window: { location: { protocol: 'http:', host: '127.0.0.1:8000' } },
        setTimeout: fn => timers.push(fn), setInterval() {},
        WebSocket: class { constructor(url) { this.url = url; sockets.push(this); } close() {} },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/js/companion.js'), 'utf8'), context);
    return { element, context, sockets, timers, run: code => vm.runInContext(code, context),
        send(data) { context.data = data; vm.runInContext('acceptCompanionSnapshot(data)', context); } };
}

function packet(overrides = {}) {
    return { type: 'companion', state: 'listening', expires_in: 8,
        message: { id: 'session:1', text: '我好難過', reply: '你希望我陪你待一下嗎？',
            intent: 'clarify', received_at: new Date().toISOString() }, ...overrides };
}

test('reply renders without face records and keeps untrusted text literal', () => {
    const p = page(), data = packet();
    data.message.text = '<img src=x onerror=alert(1)>';
    p.send(data);
    assert.equal(p.element('companion-transcript').textContent, data.message.text);
    assert.equal(p.element('companion-transcript').innerHTML, '');
    assert.equal(p.element('companion-reply').textContent, data.message.reply);
    assert.equal(p.element('companion-voice-status').dataset.state, 'listening');
    assert.equal(p.element('companion-reply-time').hidden, false);
});

test('duplicate message is not replaced or reannounced', () => {
    const p = page(), data = packet(); p.send(data);
    data.message.reply = 'duplicate should not overwrite'; p.send(data);
    assert.equal(p.element('companion-reply').textContent, '你希望我陪你待一下嗎？');
    data.message.id = 'session:2'; p.send(data);
    assert.equal(p.element('companion-reply').textContent, data.message.reply);
});

test('source expiry clears text even if no more socket packets arrive', () => {
    const p = page(); p.send(packet());
    p.run('renderCompanionDialogue(Date.now() + 9000)');
    assert.equal(p.element('companion-voice-status').dataset.state, 'stale');
    assert.equal(p.element('companion-reply').textContent, '等待對話');
    assert.equal(p.element('companion-reply-time').hidden, true);
});

test('stopping, reply expiry, and language toggle preserve separation', () => {
    const p = page(); p.send(packet());
    p.run("currentLang = 'en'; renderCompanionDialogue()");
    assert.equal(p.element('companion-voice-status').textContent, 'Laptop microphone active');
    assert.equal(p.element('companion-reply').textContent, '你希望我陪你待一下嗎？');
    p.send(packet({ message: null }));
    assert.equal(p.element('companion-reply').textContent, 'Waiting for conversation');
    p.send(packet({ state: 'stopped', expires_in: 0, message: null }));
    assert.equal(p.element('companion-voice-status').textContent, 'Laptop microphone stopped');
});

test('invalid packets and emotion records do not change dialogue', () => {
    const p = page(); p.send(packet());
    for (const data of [null, [], { id: 1, fused_emotion: 'happy' },
        packet({ state: 'invented' }), packet({ expires_in: 999 }), packet({ message: {} })]) p.send(data);
    assert.equal(p.element('companion-reply').textContent, '你希望我陪你待一下嗎？');
});

test('dedicated socket reconnects, clears old text, and ignores obsolete sockets', () => {
    const p = page(); p.run('connectCompanionDialogue()');
    const first = p.sockets[0];
    assert.equal(first.url, 'ws://127.0.0.1:8000/ws/companion');
    first.onopen(); first.onmessage({ data: JSON.stringify(packet()) });
    first.onclose({ code: 1006 });
    assert.equal(p.element('companion-voice-status').dataset.state, 'disconnected');
    assert.equal(p.element('companion-reply').textContent, '等待對話');
    p.timers[0](); p.sockets[1].onopen();
    first.onmessage({ data: JSON.stringify(packet()) });
    assert.equal(p.element('companion-reply').textContent, '等待對話');
    p.sockets[1].onmessage({ data: JSON.stringify(packet()) });
    assert.equal(p.element('companion-reply').textContent, '你希望我陪你待一下嗎？');
});

test('disabled feature stops retries and shows an explicit state', () => {
    const p = page(); p.run('connectCompanionDialogue()');
    p.sockets[0].onopen(); p.sockets[0].onclose({ code: 4404 });
    assert.equal(p.element('companion-voice-status').dataset.state, 'disabled');
    assert.equal(p.timers.length, 0);
});
