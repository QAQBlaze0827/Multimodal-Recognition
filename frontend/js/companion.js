// Text replies have their own connection and never enter the emotion pipeline.
let companionSocket = null;
let companionVoiceState = 'connecting';
let companionVoiceDeadline = 0;
let companionReply = null;
let companionLastMessageId = null;
// Convert only displayed transcripts; keep incoming text and reply decisions intact.
const companionToTraditional = typeof OpenCC !== 'undefined'
    ? OpenCC.Converter({ from: 'cn', to: 'tw' }) : value => value;
if (typeof OpenCC === 'undefined') console.warn('Transcript converter unavailable; showing original text.');

const COMPANION_TEXT = {
    zh: {
        voice: '語音互動', reply: '柴犬回覆', user: '你的話（轉錄）', face: '表情互動：',
        emptyReply: '等待對話', emptyText: '尚未收到文字', latest: '最近一次回覆',
        connecting: '連線中...', waiting: '等待筆電轉錄程式', listening: '筆電收音中',
        stopped: '筆電收音已停止', stale: '轉錄程式連線逾時',
        disconnected: '語音網頁連線中斷', disabled: '語音網頁連線未啟用',
    },
    en: {
        voice: 'Voice interaction', reply: 'Shiba reply', user: 'Your words (transcribed)', face: 'Facial interaction: ',
        emptyReply: 'Waiting for conversation', emptyText: 'No transcript yet', latest: 'Last reply',
        connecting: 'Connecting...', waiting: 'Waiting for laptop transcription', listening: 'Laptop microphone active',
        stopped: 'Laptop microphone stopped', stale: 'Transcription connection timed out',
        disconnected: 'Voice bridge disconnected', disabled: 'Voice bridge not enabled',
    },
};

function acceptCompanionSnapshot(data, now = Date.now()) {
    if (!data || data.type !== 'companion' ||
            !['waiting', 'listening', 'stopped', 'stale'].includes(data.state) ||
            !Number.isFinite(data.expires_in) || data.expires_in < 0 || data.expires_in > 8) return;
    const message = data.message;
    if (message !== null && (!message || typeof message.id !== 'string' ||
            typeof message.text !== 'string' || !message.text.trim() || message.text.length > 2000 ||
            typeof message.reply !== 'string' || !message.reply.trim() || message.reply.length > 500 ||
            !['company', 'play', 'rest', 'quiet', 'clarify'].includes(message.intent) ||
            !Number.isFinite(Date.parse(message.received_at)))) return;
    companionVoiceState = data.state;
    companionVoiceDeadline = now + data.expires_in * 1000;
    if (data.state !== 'listening' || message === null) {
        companionReply = null;
    } else if (message.id !== companionLastMessageId) {
        companionReply = { ...message };
        companionLastMessageId = message.id;
    }
    renderCompanionDialogue(now);
}

function renderCompanionDialogue(now = Date.now()) {
    const labels = COMPANION_TEXT[currentLang] || COMPANION_TEXT.zh;
    if (companionVoiceState === 'listening' && now >= companionVoiceDeadline) {
        companionVoiceState = 'stale';
        companionReply = null;
    }
    const text = (id, value) => {
        const element = document.getElementById(id);
        if (element && element.textContent !== value) element.textContent = value;
    };
    text('companion-voice-label', labels.voice);
    text('companion-reply-label', labels.reply);
    text('companion-user-label', labels.user);
    text('companion-face-label', labels.face);
    text('companion-voice-status', labels[companionVoiceState]);
    document.getElementById('companion-voice-status').dataset.state = companionVoiceState;
    text('companion-reply', companionReply ? companionReply.reply : labels.emptyReply);
    text('companion-transcript', companionReply ? companionToTraditional(companionReply.text) : labels.emptyText);
    const stamp = document.getElementById('companion-reply-time');
    stamp.hidden = !companionReply;
    if (companionReply) {
        stamp.dateTime = companionReply.received_at;
        text('companion-reply-time', `${labels.latest} · ${new Date(companionReply.received_at)
            .toLocaleTimeString(currentLang === 'en' ? 'en-GB' : 'zh-TW')}`);
    } else {
        stamp.dateTime = '';
        text('companion-reply-time', '');
    }
}

function connectCompanionDialogue() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${proto}//${window.location.host}/ws/companion`);
    companionSocket = socket;
    socket.onopen = () => {
        if (companionSocket !== socket) return;
        companionVoiceState = 'waiting';
        companionReply = null;
        companionLastMessageId = null;
        renderCompanionDialogue();
    };
    socket.onmessage = event => {
        if (companionSocket !== socket) return;
        try { acceptCompanionSnapshot(JSON.parse(event.data)); }
        catch { /* Ignore malformed events without displaying untrusted markup. */ }
    };
    socket.onclose = event => {
        if (companionSocket !== socket) return;
        companionVoiceState = event.code === 4404 ? 'disabled' : 'disconnected';
        companionReply = null;
        companionLastMessageId = null;
        renderCompanionDialogue();
        if (event.code !== 4404) setTimeout(() => {
            if (companionSocket === socket) connectCompanionDialogue();
        }, 3000);
    };
    socket.onerror = () => socket.close();
}

document.addEventListener('DOMContentLoaded', () => {
    renderCompanionDialogue();
    connectCompanionDialogue();
    setInterval(renderCompanionDialogue, 1000);
});
