let currentView = 'live';
let replayData = [];
let replayIndex = 0;
let replayTimer = null;
let replaySpeed = 1;

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    initLiveView();
    initNav();
    initHistoryView();
    initAnalyticsView();
    initReplayView();

    onWsMessage(data => {
        if (currentView === 'live') {
            updateLiveView(data);
        }
    });
});

function initNav() {
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const view = link.dataset.view;
            switchView(view);
        });
    });
}

function switchView(view) {
    currentView = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');
    document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
    document.querySelector(`nav a[data-view="${view}"]`).classList.add('active');

    if (view === 'history') loadHistory();
    if (view === 'analytics') loadAnalytics();
    if (view === 'replay') loadReplaySessions();
}

function initLiveView() {
    getRecentLogs(1).then(logs => {
        if (logs.length > 0) updateLiveView(logs[0]);
    }).catch(() => {});
}

function updateLiveView(data) {
    updateEmotionDisplay('fused', data.fused_emotion, data.fused_conf, 'Fused');
    updateEmotionDisplay('video', data.video_emotion || '-', data.video_conf || 0, 'Video');
    updateEmotionDisplay('audio', data.audio_emotion || '-', data.audio_conf || 0, 'Audio');

    const scores = readScores(data, 'video');
    renderBars('live-bars', scores, data.fused_emotion);

    document.getElementById('live-fps').textContent = data.fps ? data.fps.toFixed(1) : '-';
    document.getElementById('live-temp').textContent = data.cpu_temp ? `${data.cpu_temp.toFixed(1)}°C` : '-';
    document.getElementById('live-face').textContent = data.face_detected ? 'Yes' : 'No';

    const dot = document.querySelector('header .dot');
    if (dot) dot.style.background = EMOTION_COLORS[data.fused_emotion] || 'var(--success)';
}

function readScores(data, source) {
    const objectKey = `${source}_scores`;
    const jsonKey = `${source}_scores_json`;
    if (data[objectKey] && typeof data[objectKey] === 'object') return data[objectKey];
    if (!data[jsonKey]) return {};
    try {
        return JSON.parse(data[jsonKey]);
    } catch (e) {
        console.warn(`Invalid ${jsonKey}:`, e);
        return {};
    }
}

function updateEmotionDisplay(prefix, emotion, confidence, source) {
    const labelEl = document.getElementById(`${prefix}-label`);
    const confEl = document.getElementById(`${prefix}-conf`);
    const srcEl = document.getElementById(`${prefix}-src`);
    if (!labelEl) return;
    labelEl.textContent = emotion;
    labelEl.style.color = EMOTION_COLORS[emotion] || '#e8eaed';
    if (confEl) confEl.textContent = `${(confidence * 100).toFixed(0)}%`;
    if (srcEl) srcEl.textContent = source;
}

function renderBars(containerId, scores, highlightEmotion) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const allEmotions = ['neutral', 'happy', 'sad', 'anger', 'fear', 'surprise', 'disgust'];
    const data = allEmotions.map(e => ({ emotion: e, confidence: scores[e] || 0 }));
    const maxConf = Math.max(...data.map(d => d.confidence), 0.01);

    container.innerHTML = data.map(d => `
        <div class="emotion-bar-container">
            <div class="emotion-bar-label">
                <span>${d.emotion}</span>
                <span>${(d.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="emotion-bar">
                <div class="fill" style="width: ${(d.confidence / maxConf) * 100}%; background: ${EMOTION_COLORS[d.emotion] || '#666'};"></div>
            </div>
        </div>
    `).join('');
}

function initHistoryView() {
    document.getElementById('history-filter-btn').addEventListener('click', loadHistory);
    document.getElementById('history-refresh').addEventListener('click', loadHistory);
}

async function loadHistory() {
    const emotion = document.getElementById('filter-emotion').value;
    const session = document.getElementById('filter-session').value;

    try {
        const logs = await getLogs({ emotion, session_id: session, limit: 500 });
        const tbody = document.querySelector('#history-table tbody');
        if (!tbody) return;

        if (logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5">No records found.</td></tr>';
            return;
        }

        tbody.innerHTML = logs.map(log => `
            <tr>
                <td>${log.timestamp || '-'}</td>
                <td><span class="emotion-badge"><span class="dot" style="background: ${EMOTION_COLORS[log.fused_emotion] || '#666'};"></span>${log.fused_emotion}</span></td>
                <td>${(log.fused_conf * 100).toFixed(0)}%</td>
                <td>${log.video_emotion || '-'} / ${log.audio_emotion || '-'}</td>
                <td>${log.fps ? log.fps.toFixed(1) : '-'}</td>
            </tr>
        `).join('');

        await populateSessionFilter();
    } catch (e) {
        console.error('History load error:', e);
    }
}

async function populateSessionFilter() {
    try {
        const sessions = await getSessions();
        const select = document.getElementById('filter-session');
        const current = select.value;
        select.innerHTML = '<option value="">All Sessions</option>' +
            sessions.map(s => `<option value="${s.id}">${s.id} (${s.log_count || 0} logs)</option>`).join('');
        select.value = current;
    } catch (e) { /* ignore */ }
}

function initAnalyticsView() {
    document.getElementById('analytics-days').addEventListener('change', loadAnalytics);
}

async function loadAnalytics() {
    const days = parseInt(document.getElementById('analytics-days').value) || 30;
    try {
        const data = await getAnalytics(days);
        if (!data.distribution || Object.keys(data.distribution).length === 0) {
            document.getElementById('analytics-summary').innerHTML = '<p>No data for this period.</p>';
            return;
        }
        createPieChart('analytics-pie', data.distribution);
        document.getElementById('analytics-summary').innerHTML = `
            <div class="grid-3">
                <div class="stat-box"><div class="value">${data.total_logs}</div><div class="label">Total Records</div></div>
                <div class="stat-box"><div class="value">${data.session_count}</div><div class="label">Sessions</div></div>
                <div class="stat-box"><div class="value">${days}d</div><div class="label">Period</div></div>
            </div>
        `;
    } catch (e) {
        console.error('Analytics load error:', e);
    }
}

function initReplayView() {
    document.getElementById('replay-load').addEventListener('click', startReplay);
    document.getElementById('replay-play').addEventListener('click', toggleReplayPlay);
    document.getElementById('replay-speed').addEventListener('change', (e) => {
        replaySpeed = parseFloat(e.target.value);
    });
    document.getElementById('replay-timeline').addEventListener('input', (e) => {
        if (replayData.length === 0) return;
        const idx = parseInt(e.target.value);
        seekReplay(idx);
    });
}

async function loadReplaySessions() {
    try {
        const sessions = await getSessions();
        const select = document.getElementById('replay-session');
        select.innerHTML = sessions.map(s =>
            `<option value="${s.id}">${s.id} (${s.log_count || 0} logs)</option>`
        ).join('');
    } catch (e) {
        console.error('Replay sessions load error:', e);
    }
}

async function startReplay() {
    const sessionId = document.getElementById('replay-session').value;
    if (!sessionId) return;

    try {
        replayData = await getSessionTimeline(sessionId);
        if (replayData.length === 0) {
            document.getElementById('replay-status').textContent = 'No data for this session.';
            return;
        }
        replayIndex = 0;
        const slider = document.getElementById('replay-timeline');
        slider.max = replayData.length - 1;
        slider.value = 0;
        document.getElementById('replay-play').disabled = false;
        document.getElementById('replay-play').textContent = '▶ Play';
        seekReplay(0);
        if (replayChart) {
            const firstScores = readScores(replayData[0], 'video');
            updateReplayChart(firstScores, replayData[0].fused_emotion);
        }
    } catch (e) {
        console.error('Replay start error:', e);
    }
}

function toggleReplayPlay() {
    const btn = document.getElementById('replay-play');
    if (replayTimer) {
        clearInterval(replayTimer);
        replayTimer = null;
        btn.textContent = '▶ Play';
    } else {
        btn.textContent = '⏸ Pause';
        const interval = Math.max(50, Math.floor(1000 / replaySpeed));
        replayTimer = setInterval(() => {
            if (replayIndex < replayData.length - 1) {
                replayIndex++;
                seekReplay(replayIndex);
                document.getElementById('replay-timeline').value = replayIndex;
            } else {
                clearInterval(replayTimer);
                replayTimer = null;
                document.getElementById('replay-play').textContent = '▶ Play';
            }
        }, interval);
    }
}

function seekReplay(idx) {
    if (idx < 0 || idx >= replayData.length) return;
    replayIndex = idx;
    const d = replayData[idx];
    document.getElementById('replay-current-idx').textContent = `${idx + 1} / ${replayData.length}`;
    document.getElementById('replay-ts').textContent = d.timestamp || '';
    document.getElementById('replay-emotion').textContent = d.fused_emotion;
    document.getElementById('replay-emotion').style.color = EMOTION_COLORS[d.fused_emotion] || '#e8eaed';
    document.getElementById('replay-conf').textContent = `${(d.fused_conf * 100).toFixed(0)}%`;
    const slider = document.getElementById('replay-timeline');
    if (slider) slider.value = idx;

    const scores = readScores(d, 'video');
    updateReplayChart(scores, d.fused_emotion);
}

function updateReplayChart(scores, highlight) {
    if (!replayChart) {
        replayChart = createReplayChart('replay-chart');
        if (!replayChart) return;
    }
    const allEmotions = ['neutral', 'happy', 'sad', 'anger', 'fear', 'surprise', 'disgust'];
    replayChart.data.datasets[0].data = allEmotions.map(e => scores[e] || 0);
    replayChart.data.datasets[0].backgroundColor = allEmotions.map(e =>
        e === highlight ? (EMOTION_COLORS[e] || '#666') : '#333'
    );
    replayChart.update();
}
