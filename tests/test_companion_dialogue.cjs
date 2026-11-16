const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function page(withConverter = true) {
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
    if (withConverter) {
        vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/vendor/opencc-js-1.4.2/cn2t.js'), 'utf8'), context);
    }
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/js/companion.js'), 'utf8'), context);
    return { element, context, sockets, timers, run: code => vm.runInContext(code, context),
        send(data) { context.data = data; vm.runInContext('acceptCompanionSnapshot(data)', context); } };
}

function packet(overrides = {}) {
    return { type: 'companion', state: 'listening', expires_in: 8,
        message: { id: 'session:1', text: '我好難過', reply: '你希望我陪你待一下嗎？',
            intent: 'clarify', received_at: new Date().toISOString() }, ...overrides };
}

test('pinned local converter is intact and loads before the dialogue script', () => {
    const vendor = 'vendor/opencc-js-1.4.2/';
    const root = path.join(__dirname, '../frontend');
    const bundle = fs.readFileSync(path.join(root, vendor, 'cn2t.js'));
    assert.equal(createHash('sha256').update(bundle).digest('hex'),
        'd66d920069ab526db08c6f2bc15fcf6feb7d07de3cf98e7e27c0aaf7d136baee');
    for (const name of ['LICENSE', 'LICENSES/Apache-2.0.txt', 'THIRD_PARTY_LICENSES.md']) {
        assert.ok(fs.existsSync(path.join(root, vendor, name)));
    }
    const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
    const converter = html.indexOf(`<script src="${vendor}cn2t.js"></script>`);
    assert.ok(converter >= 0 && converter < html.indexOf('<script src="js/companion.js"></script>'));
});

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

test('simplified transcript displays as traditional without changing source or reply', () => {
    const p = page(), data = packet();
    data.message.text = '我好难过，可以陪我玩吗？';
    data.message.reply = '原始回覆不转字';
    p.send(data);
    assert.equal(p.element('companion-transcript').textContent, '我好難過，可以陪我玩嗎？');
    assert.equal(p.run('companionReply.text'), '我好难过，可以陪我玩吗？');
    assert.equal(data.message.text, '我好难过，可以陪我玩吗？');
    assert.equal(p.run('companionReply.intent'), 'clarify');
    assert.equal(p.element('companion-reply').textContent, '原始回覆不转字');
});

test('traditional text, mixed English, punctuation and ASR mistakes are preserved', () => {
    const p = page();
    const cases = [
        ['我現在不想玩，請先安靜。', '我現在不想玩，請先安靜。'],
        ['Hello 123！我好难过\n可以陪我吗？', 'Hello 123！我好難過\n可以陪我嗎？'],
        ['我今天有點內想休息一下', '我今天有點內想休息一下'],
        ['Please wait, 123.', 'Please wait, 123.'],
        ['我想理发，然后出发', '我想理髮，然後出發'],
    ];
    cases.forEach(([input, expected], index) => {
        const data = packet();
        data.message.id = `session:${index}`;
        data.message.text = input;
        p.send(data);
        assert.equal(p.element('companion-transcript').textContent, expected);
        assert.equal(p.run('companionReply.text'), input);
    });
});

test('language toggle changes labels, not the traditional transcript or raw text', () => {
    const p = page(), data = packet();
    data.message.text = '请先安静，我现在不想玩';
    p.send(data);
    p.run("currentLang = 'en'; renderCompanionDialogue()");
    assert.equal(p.element('companion-user-label').textContent, 'Your words (transcribed)');
    assert.equal(p.element('companion-transcript').textContent, '請先安靜，我現在不想玩');
    assert.equal(p.run('companionReply.text'), data.message.text);
    p.run("currentLang = 'zh'; renderCompanionDialogue()");
    assert.equal(p.element('companion-transcript').textContent, '請先安靜，我現在不想玩');
    p.send(packet({ state: 'stopped', expires_in: 0, message: null }));
    assert.equal(p.element('companion-transcript').textContent, '尚未收到文字');
});

test('converted transcript still treats HTML as text', () => {
    const p = page(), data = packet();
    data.message.text = '<img src=x onerror=alert(1)>我想说话';
    p.send(data);
    assert.equal(p.element('companion-transcript').textContent, '<img src=x onerror=alert(1)>我想說話');
    assert.equal(p.element('companion-transcript').innerHTML, '');
});

test('missing converter falls back to original text without breaking dialogue', () => {
    const p = page(false), data = packet();
    data.message.text = '我好难过';
    p.send(data);
    assert.equal(p.element('companion-transcript').textContent, data.message.text);
    assert.equal(p.element('companion-reply').textContent, data.message.reply);
    assert.equal(p.element('companion-voice-status').dataset.state, 'listening');
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
