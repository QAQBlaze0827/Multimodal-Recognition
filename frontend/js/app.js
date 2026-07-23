let currentView = 'live';
let replayData = [];
let replayIndex = 0;
let replayTimer = null;
let replaySpeed = 1;
let liveTrend = [];
let lastLiveData = null;
const TREND_LIMIT = 12;
let currentLang = localStorage.getItem('uiLang') || 'zh';
window.APP_LANG = currentLang;

const UI_TEXT = {
    zh: {
        appTitle: '多模態情緒辨識儀表板',
        pageTitle: '多模態情緒辨識儀表板',
        wsLabel: '連線：',
        langButton: 'EN',
        navLive: '即時狀態',
        navReplay: '回放',
        navHistory: '歷史紀錄',
        navAnalytics: '統計分析',
        connected: '已連線',
        disconnected: '已中斷',
        connecting: '連線中...',
        currentReading: '目前判讀',
        waitingData: '等待資料',
        startPrompt: '啟動辨識程式後，這裡會顯示即時情緒狀態。',
        noSignal: '尚無訊號',
        noAudioLabel: '--',
        noAudioSignal: '尚無語音訊號',
        noTrend: '尚無趨勢',
        fused: '綜合結果',
        video: '臉部表情',
        audio: '語音情緒',
        fusedSource: '綜合',
        videoSource: '視覺',
        audioSource: '音訊',
        emotionScores: '情緒分數',
        signalQuality: '判斷依據',
        face: '臉部',
        voice: '語音',
        agreement: '一致性',
        waiting: '等待中',
        recentTrend: '最近趨勢',
        trendPrompt: '累積幾筆資料後，會顯示最近情緒變化。',
        noLiveReadings: '尚無即時資料。',
        systemStatus: '系統狀態',
        temperature: '溫度',
        faceDetected: '臉部偵測',
        yes: '有',
        no: '無',
        sessionReplay: 'Session 回放',
        loadingSessions: '載入 session 中...',
        load: '載入',
        speed: '速度：',
        replayInsight: '載入 session 後，可逐筆查看當下情緒判讀。',
        frameScores: '當下分數',
        historyTitle: '辨識歷史',
        allEmotions: '全部情緒',
        allSessions: '全部 Session',
        filter: '篩選',
        refresh: '重新整理',
        historyPrompt: '選擇篩選條件或重新整理，即可查看最近紀錄摘要。',
        loading: '載入中...',
        timestamp: '時間',
        emotion: '情緒',
        confidence: '可信度',
        videoAudio: '視覺/音訊',
        analyticsTitle: '情緒分布',
        period: '期間：',
        sevenDays: '最近 7 天',
        thirtyDays: '最近 30 天',
        ninetyDays: '最近 90 天',
        year: '最近一年',
        analyticsPrompt: '選擇期間後，這裡會整理主要情緒分布。',
        play: '播放',
        pause: '暫停',
    },
    en: {
        appTitle: 'Multimodal Emotion Dashboard',
        pageTitle: 'Multimodal Emotion Dashboard',
        wsLabel: 'Connection: ',
        langButton: '中文',
        navLive: 'Live',
        navReplay: 'Replay',
        navHistory: 'History',
        navAnalytics: 'Analytics',
        connected: 'connected',
        disconnected: 'disconnected',
        connecting: 'connecting...',
        currentReading: 'Current Reading',
        waitingData: 'Waiting for data',
        startPrompt: 'Start the recognition process to see live emotion feedback.',
        noSignal: 'No signal',
        noAudioLabel: '--',
        noAudioSignal: 'No audio signal',
        noTrend: 'No trend yet',
        fused: 'Fused',
        video: 'Video',
        audio: 'Audio',
        fusedSource: 'Fused',
        videoSource: 'Video',
        audioSource: 'Audio',
        emotionScores: 'Emotion Scores',
        signalQuality: 'Signal Quality',
        face: 'Face',
        voice: 'Voice',
        agreement: 'Agreement',
        waiting: 'Waiting',
        recentTrend: 'Recent Trend',
        trendPrompt: 'After a few readings, recent emotion changes will appear here.',
        noLiveReadings: 'No live readings yet.',
        systemStatus: 'System Status',
        temperature: 'Temperature',
        faceDetected: 'Face Detected',
        yes: 'Yes',
        no: 'No',
        sessionReplay: 'Session Replay',
        loadingSessions: 'Loading sessions...',
        load: 'Load',
        speed: 'Speed: ',
        replayInsight: 'Load a session to inspect each moment.',
        frameScores: 'Frame Scores',
        historyTitle: 'Recognition History',
        allEmotions: 'All Emotions',
        allSessions: 'All Sessions',
        filter: 'Filter',
        refresh: 'Refresh',
        historyPrompt: 'Choose a filter or refresh to review recent records.',
        loading: 'Loading...',
        timestamp: 'Timestamp',
        emotion: 'Emotion',
        confidence: 'Confidence',
        videoAudio: 'Video/Audio',
        analyticsTitle: 'Emotion Distribution',
        period: 'Period: ',
        sevenDays: 'Last 7 days',
        thirtyDays: 'Last 30 days',
        ninetyDays: 'Last 90 days',
        year: 'Last year',
        analyticsPrompt: 'Select a period to summarize detected emotions.',
        play: 'Play',
        pause: 'Pause',
    },
};

document.addEventListener('DOMContentLoaded', () => {
    applyLanguage();
    initLanguageToggle();
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

function tr(key) {
    return UI_TEXT[currentLang][key] || UI_TEXT.en[key] || key;
}

function setText(selector, key) {
    const el = document.querySelector(selector);
    if (el) el.textContent = tr(key);
}

function initLanguageToggle() {
    const btn = document.getElementById('lang-toggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
        currentLang = currentLang === 'zh' ? 'en' : 'zh';
        window.APP_LANG = currentLang;
        localStorage.setItem('uiLang', currentLang);
        applyLanguage();
    });
}

function applyLanguage() {
    document.documentElement.lang = currentLang === 'zh' ? 'zh-Hant' : 'en';
    document.title = tr('pageTitle');
    setText('#app-title', 'appTitle');
    setText('#ws-label', 'wsLabel');
    setText('#lang-toggle', 'langButton');
    setText('nav a[data-view="live"]', 'navLive');
    setText('nav a[data-view="replay"]', 'navReplay');
    setText('nav a[data-view="history"]', 'navHistory');
    setText('nav a[data-view="analytics"]', 'navAnalytics');
    const liveHeadings = document.querySelectorAll('#view-live h2');
    ['currentReading', 'fused', 'video', 'audio', 'emotionScores', 'signalQuality', 'recentTrend', 'systemStatus']
        .forEach((key, index) => {
            if (liveHeadings[index]) liveHeadings[index].textContent = tr(key);
        });
    const insightLabels = document.querySelectorAll('#view-live .insight-list span');
    ['face', 'voice', 'agreement'].forEach((key, index) => {
        if (insightLabels[index]) insightLabels[index].textContent = tr(key);
    });
    const fpsLabel = document.querySelector('#live-fps + .label');
    const tempLabel = document.querySelector('#live-temp + .label');
    const faceLabel = document.querySelector('#live-face + .label');
    if (fpsLabel) fpsLabel.textContent = 'FPS';
    if (tempLabel) tempLabel.textContent = tr('temperature');
    if (faceLabel) faceLabel.textContent = tr('faceDetected');
    setText('#view-history .card > h2', 'historyTitle');
    setText('#view-analytics .card > h2', 'analyticsTitle');
    setText('#insight-face', 'waiting');
    setText('#insight-audio', 'waiting');
    setText('#insight-agreement', 'waiting');
    setText('#replay-load', 'load');
    setText('#replay-play', replayTimer ? 'pause' : 'play');
    setText('#history-filter-btn', 'filter');
    setText('#history-refresh', 'refresh');
    setText('#history-summary', 'historyPrompt');
    setText('#analytics-insight', 'analyticsPrompt');

    const staticLabels = [
        ['#summary-emotion', 'waitingData'],
        ['#summary-line', 'startPrompt'],
        ['#summary-confidence', 'noSignal'],
        ['#summary-stability', 'noTrend'],
        ['#trend-summary', 'trendPrompt'],
    ];
    if (!lastLiveData) staticLabels.forEach(([selector, key]) => setText(selector, key));

    updateStaticFormLabels();
    updateTableHeaders();
    updateReplayStaticLabels();
    updateStatusLanguage();
    if (lastLiveData) updateLiveView(lastLiveData);
    if (liveTrend.length) updateLiveTrend(liveTrend[liveTrend.length - 1], false);
    if (currentView === 'history') loadHistory();
    if (currentView === 'analytics') loadAnalytics();
    if (replayData.length) {
        if (replayChart) {
            replayChart.destroy();
            replayChart = null;
        }
        seekReplay(replayIndex);
    }
}

function updateStaticFormLabels() {
    const emotionSelect = document.getElementById('filter-emotion');
    if (emotionSelect) {
        const labels = ['', 'neutral', 'happy', 'sad', 'anger'];
        [...emotionSelect.options].forEach((option, index) => {
            option.textContent = index === 0 ? tr('allEmotions') : formatEmotion(labels[index]);
        });
    }
    const filterSession = document.getElementById('filter-session');
    if (filterSession && filterSession.options.length) filterSession.options[0].textContent = tr('allSessions');
    const replaySession = document.getElementById('replay-session');
    if (replaySession && replaySession.options.length && !replaySession.value) {
        replaySession.options[0].textContent = tr('loadingSessions');
    }
    const periodLabel = document.querySelector('#view-analytics .filter-bar label');
    if (periodLabel && periodLabel.firstChild) periodLabel.firstChild.textContent = tr('period');
    const days = document.getElementById('analytics-days');
    if (days) {
        const keys = ['sevenDays', 'thirtyDays', 'ninetyDays', 'year'];
        [...days.options].forEach((option, index) => {
            option.textContent = tr(keys[index]);
        });
    }
}

function updateTableHeaders() {
    const headers = document.querySelectorAll('#history-table th');
    const keys = ['timestamp', 'emotion', 'confidence', 'videoAudio', 'FPS'];
    headers.forEach((th, index) => {
        th.textContent = keys[index] === 'FPS' ? 'FPS' : tr(keys[index]);
    });
}

function updateReplayStaticLabels() {
    setText('#view-replay .card:nth-child(1) h2', 'sessionReplay');
    setText('#view-replay .card:nth-child(3) h2', 'frameScores');
    const speedLabel = document.querySelector('.replay-controls label');
    if (speedLabel && speedLabel.firstChild) speedLabel.firstChild.textContent = tr('speed');
    if (!replayData.length) setText('#replay-insight', 'replayInsight');
}

function updateStatusLanguage() {
    const status = document.getElementById('ws-status');
    if (!status) return;
    if (status.dataset.state === 'connected') status.textContent = tr('connected');
    if (status.dataset.state === 'disconnected') status.textContent = tr('disconnected');
    if (status.dataset.state === 'connecting') status.textContent = tr('connecting');
}

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
    lastLiveData = data;
    updateEmotionDisplay('fused', data.fused_emotion, data.fused_conf, tr('fusedSource'));
    updateEmotionDisplay('video', data.video_emotion || '-', data.video_conf || 0, tr('videoSource'));
    updateEmotionDisplay('audio', data.audio_emotion || '-', data.audio_conf || 0, tr('audioSource'));

    const scores = readScores(data, 'video');
    renderBars('live-bars', scores, data.fused_emotion);
    updateLiveSummary(data);
    updateSignalQuality(data);
    updateLiveTrend(data);

    document.getElementById('live-fps').textContent = data.fps ? data.fps.toFixed(1) : '-';
    document.getElementById('live-temp').textContent = data.cpu_temp ? `${data.cpu_temp.toFixed(1)}°C` : '-';
    document.getElementById('live-face').textContent = data.face_detected ? '有' : '無';

    const dot = document.querySelector('header .dot');
    if (dot) dot.style.background = EMOTION_COLORS[data.fused_emotion] || 'var(--success)';
}

function updateLiveSummary(data) {
    const emotion = data.fused_emotion || 'unknown';
    const confidence = Number(data.fused_conf || 0);
    const summaryEmotion = document.getElementById('summary-emotion');
    const summaryLine = document.getElementById('summary-line');
    const confidencePill = document.getElementById('summary-confidence');
    const stabilityPill = document.getElementById('summary-stability');
    if (!summaryEmotion || !summaryLine) return;

    summaryEmotion.textContent = formatEmotion(emotion);
    summaryEmotion.style.color = EMOTION_COLORS[emotion] || '#e8eaed';
    summaryLine.textContent = buildReadingSentence(data);
    if (confidencePill) {
        const level = confidenceLevel(confidence);
        confidencePill.textContent = `${level}（${Math.round(confidence * 100)}%）`;
        confidencePill.dataset.level = confidenceLevelKey(confidence);
    }
    if (stabilityPill) stabilityPill.textContent = trendStabilityLabel();
}

function updateSignalQuality(data) {
    const face = document.getElementById('insight-face');
    const audio = document.getElementById('insight-audio');
    const agreement = document.getElementById('insight-agreement');
    if (face) face.textContent = data.face_detected ? `已偵測（${data.face_backend || 'camera'}）` : '未偵測到臉部';
    if (audio) {
        const conf = Number(data.audio_conf || 0);
        audio.textContent = hasAudioSignal(data) ? `${formatEmotion(data.audio_emotion)} ${Math.round(conf * 100)}%` : tr('noAudioSignal');
    }
    if (agreement) agreement.textContent = agreementLabel(data);
}

function updateLiveTrend(data, append = true) {
    if (!data.fused_emotion) return;
    if (append) {
        liveTrend.push({
            emotion: data.fused_emotion,
            confidence: Number(data.fused_conf || 0),
            timestamp: data.timestamp || '',
        });
        if (liveTrend.length > TREND_LIMIT) liveTrend = liveTrend.slice(-TREND_LIMIT);
    }

    const container = document.getElementById('live-trend');
    if (!container) return;
    updateTrendSummary();
    container.innerHTML = liveTrend.map(item => `
        <span class="trend-chip" title="${item.timestamp} ${Math.round(item.confidence * 100)}%" style="background: ${EMOTION_COLORS[item.emotion] || '#666'}">
            ${shortEmotion(item.emotion)}
        </span>
    `).join('');
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

function formatEmotion(emotion) {
    if (!emotion || emotion === '-') return '-';
    const zhLabels = {
        neutral: '中性',
        happy: '開心',
        sad: '悲傷',
        anger: '生氣',
        unknown: '未知',
    };
    const enLabels = {
        neutral: 'Neutral',
        happy: 'Happy',
        sad: 'Sad',
        anger: 'Anger',
        unknown: 'Unknown',
    };
    const labels = currentLang === 'zh' ? zhLabels : enLabels;
    return labels[emotion] || emotion;
}

function shortEmotion(emotion) {
    const zhLabels = {
        neutral: '中',
        happy: '喜',
        sad: '悲',
        anger: '怒',
    };
    const enLabels = {
        neutral: 'Neu',
        happy: 'Hap',
        sad: 'Sad',
        anger: 'Ang',
    };
    const labels = currentLang === 'zh' ? zhLabels : enLabels;
    return labels[emotion] || '--';
}

function confidenceLevel(confidence) {
    if (currentLang === 'en') {
        if (confidence >= 0.8) return 'High confidence';
        if (confidence >= 0.5) return 'Medium confidence';
        if (confidence >= 0.3) return 'Low confidence';
        return 'Uncertain';
    }
    if (confidence >= 0.8) return '高可信';
    if (confidence >= 0.5) return '中等可信';
    if (confidence >= 0.3) return '低可信';
    return '不確定';
}

function confidenceLevelKey(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    if (confidence >= 0.3) return 'low';
    return 'uncertain';
}

function agreementLabel(data) {
    const video = data.video_emotion;
    const audio = hasAudioSignal(data) ? data.audio_emotion : '';
    if (currentLang === 'en') {
        if (!video && !audio) return 'No signal yet';
        if (!audio) return 'Video is the main signal';
        if (!video) return 'Audio is the main signal';
        if (video === audio) return 'Video and audio agree';
        return `Mixed signal: ${formatEmotion(video)} / ${formatEmotion(audio)}`;
    }
    if (!video && !audio) return '尚無訊號';
    if (!audio) return '目前主要依據視覺';
    if (!video) return '目前主要依據音訊';
    if (video === audio) return '視覺與音訊一致';
    return `判斷不一致：${formatEmotion(video)} / ${formatEmotion(audio)}`;
}

function buildReadingSentence(data) {
    const emotion = formatEmotion(data.fused_emotion || 'unknown');
    const confidence = Number(data.fused_conf || 0);
    const level = confidenceLevel(confidence);
    const agreement = agreementLabel(data);
    if (currentLang === 'en') {
        if (confidenceLevelKey(confidence) === 'uncertain') {
            return `The system is not confident yet. Current best guess is ${emotion}.`;
        }
        return `Current best guess is ${emotion}. ${level}. ${agreement}.`;
    }
    if (confidenceLevelKey(confidence) === 'uncertain') {
        return `系統目前還不太確定，暫時判斷為「${emotion}」。`;
    }
    return `目前判斷為「${emotion}」，可信度為${level}。${agreement}。`;
}

function trendStabilityLabel() {
    if (liveTrend.length < 4) return currentLang === 'zh' ? '累積趨勢中' : 'Collecting trend';
    const recent = liveTrend.slice(-6);
    const changes = recent.slice(1).filter((item, index) => item.emotion !== recent[index].emotion).length;
    if (currentLang === 'en') {
        if (changes === 0) return 'Stable trend';
        if (changes >= 3) return 'Changing quickly';
        return 'Slightly changing';
    }
    if (changes === 0) return '趨勢穩定';
    if (changes >= 3) return '變化較快';
    return '輕微變化';
}

function updateTrendSummary() {
    const el = document.getElementById('trend-summary');
    if (!el) return;
    if (liveTrend.length < 2) {
        el.textContent = '累積幾筆資料後，會顯示最近情緒變化。';
        return;
    }
    const counts = {};
    liveTrend.forEach(item => {
        counts[item.emotion] = (counts[item.emotion] || 0) + 1;
    });
    const [topEmotion, topCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    if (currentLang === 'en') {
        el.textContent = `Recent ${liveTrend.length} readings mostly show ${formatEmotion(topEmotion)} (${Math.round(topCount / liveTrend.length * 100)}%). ${trendStabilityLabel()}.`;
    } else {
        el.textContent = `最近 ${liveTrend.length} 筆主要偏向「${formatEmotion(topEmotion)}」（${Math.round(topCount / liveTrend.length * 100)}%），${trendStabilityLabel()}。`;
    }
}

function summarizeLogs(logs) {
    if (!logs.length) return null;
    const counts = {};
    let confidenceTotal = 0;
    let faceCount = 0;
    logs.forEach(log => {
        counts[log.fused_emotion] = (counts[log.fused_emotion] || 0) + 1;
        confidenceTotal += Number(log.fused_conf || 0);
        if (log.face_detected) faceCount += 1;
    });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return {
        topEmotion: top ? top[0] : '',
        topCount: top ? top[1] : 0,
        total: logs.length,
        avgConfidence: confidenceTotal / logs.length,
        faceRate: faceCount / logs.length,
    };
}

function distributionTop(distribution) {
    const entries = Object.entries(distribution || {});
    if (!entries.length) return null;
    return entries.sort((a, b) => b[1].count - a[1].count)[0];
}

function updateEmotionDisplay(prefix, emotion, confidence, source) {
    const labelEl = document.getElementById(`${prefix}-label`);
    const confEl = document.getElementById(`${prefix}-conf`);
    const srcEl = document.getElementById(`${prefix}-src`);
    if (!labelEl) return;
    if (prefix === 'audio' && (!emotion || emotion === '-' || Number(confidence || 0) <= 0)) {
        labelEl.textContent = tr('noAudioLabel');
        labelEl.style.color = '#77778a';
        if (confEl) confEl.textContent = '--';
        if (srcEl) srcEl.textContent = source;
        return;
    }
    labelEl.textContent = formatEmotion(emotion);
    labelEl.style.color = EMOTION_COLORS[emotion] || '#e8eaed';
    if (confEl) confEl.textContent = `${(confidence * 100).toFixed(0)}%`;
    if (srcEl) srcEl.textContent = source;
}

function hasAudioSignal(data) {
    return Boolean(data.audio_emotion) && Number(data.audio_conf || 0) > 0;
}

function renderBars(containerId, scores, highlightEmotion) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const allEmotions = ['neutral', 'happy', 'sad', 'anger'];
    const data = allEmotions.map(e => ({ emotion: e, confidence: scores[e] || 0 }));
    const maxConf = Math.max(...data.map(d => d.confidence), 0.01);

    container.innerHTML = data.map(d => `
        <div class="emotion-bar-container">
            <div class="emotion-bar-label">
                <span>${formatEmotion(d.emotion)}</span>
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
            tbody.innerHTML = `<tr><td colspan="5">${currentLang === 'zh' ? '目前沒有符合條件的紀錄。' : 'No records match the current filter.'}</td></tr>`;
            document.getElementById('history-summary').textContent =
                currentLang === 'zh' ? '目前沒有符合篩選條件的紀錄。' : 'No records match the current filter.';
            return;
        }

        const summary = summarizeLogs(logs);
        if (summary) {
            document.getElementById('history-summary').textContent = currentLang === 'zh'
                ? `共 ${summary.total} 筆紀錄，最常出現「${formatEmotion(summary.topEmotion)}」（${Math.round(summary.topCount / summary.total * 100)}%）。平均可信度 ${Math.round(summary.avgConfidence * 100)}%，臉部偵測成功率 ${Math.round(summary.faceRate * 100)}%。`
                : `${summary.total} records. Most common emotion is ${formatEmotion(summary.topEmotion)} (${Math.round(summary.topCount / summary.total * 100)}%). Average confidence is ${Math.round(summary.avgConfidence * 100)}%. Face was detected in ${Math.round(summary.faceRate * 100)}% of records.`;
        }

        tbody.innerHTML = logs.map(log => `
            <tr>
                <td>${log.timestamp || '-'}</td>
                <td><span class="emotion-badge"><span class="dot" style="background: ${EMOTION_COLORS[log.fused_emotion] || '#666'};"></span>${formatEmotion(log.fused_emotion)}</span></td>
                <td>${(log.fused_conf * 100).toFixed(0)}%</td>
                <td>${formatEmotion(log.video_emotion) || '-'} / ${formatEmotion(log.audio_emotion) || '-'}</td>
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
        select.innerHTML = '<option value="">全部 Session</option>' +
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
            document.getElementById('analytics-summary').innerHTML =
                `<p>${currentLang === 'zh' ? '這段期間沒有資料。' : 'No data for this period.'}</p>`;
            document.getElementById('analytics-insight').textContent =
                currentLang === 'zh' ? '選取的期間內沒有情緒紀錄。' : 'No emotion records were found in the selected period.';
            return;
        }
        createPieChart('analytics-pie', data.distribution);
        const top = distributionTop(data.distribution);
        if (top) {
            const [emotion, info] = top;
            document.getElementById('analytics-insight').textContent = currentLang === 'zh'
                ? `主要趨勢：「${formatEmotion(emotion)}」是最常出現的情緒（${info.pct}%），該情緒平均可信度 ${Math.round((info.avg_conf || 0) * 100)}%。`
                : `Main pattern: ${formatEmotion(emotion)} is the most frequent emotion (${info.pct}%). Average confidence for that emotion is ${Math.round((info.avg_conf || 0) * 100)}%.`;
        }
        document.getElementById('analytics-summary').innerHTML = `
            <div class="grid-3">
                <div class="stat-box"><div class="value">${data.total_logs}</div><div class="label">${currentLang === 'zh' ? '總紀錄' : 'Total Records'}</div></div>
                <div class="stat-box"><div class="value">${data.session_count}</div><div class="label">Sessions</div></div>
                <div class="stat-box"><div class="value">${currentLang === 'zh' ? `${days} 天` : `${days}d`}</div><div class="label">${currentLang === 'zh' ? '期間' : 'Period'}</div></div>
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
            document.getElementById('replay-status').textContent =
                currentLang === 'zh' ? '這個 session 沒有資料。' : 'No data for this session.';
            return;
        }
        replayIndex = 0;
        const slider = document.getElementById('replay-timeline');
        slider.max = replayData.length - 1;
        slider.value = 0;
        document.getElementById('replay-play').disabled = false;
        document.getElementById('replay-play').textContent = '播放';
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
        btn.textContent = '播放';
    } else {
        btn.textContent = '暫停';
        const interval = Math.max(50, Math.floor(1000 / replaySpeed));
        replayTimer = setInterval(() => {
            if (replayIndex < replayData.length - 1) {
                replayIndex++;
                seekReplay(replayIndex);
                document.getElementById('replay-timeline').value = replayIndex;
            } else {
                clearInterval(replayTimer);
                replayTimer = null;
                document.getElementById('replay-play').textContent = '播放';
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
    document.getElementById('replay-emotion').textContent = formatEmotion(d.fused_emotion);
    document.getElementById('replay-emotion').style.color = EMOTION_COLORS[d.fused_emotion] || '#e8eaed';
    document.getElementById('replay-conf').textContent = `${(d.fused_conf * 100).toFixed(0)}%`;
    document.getElementById('replay-insight').textContent = currentLang === 'zh'
        ? `此刻判斷為「${formatEmotion(d.fused_emotion)}」，可信度為${confidenceLevel(Number(d.fused_conf || 0))}。`
        : `${formatEmotion(d.fused_emotion)} at ${confidenceLevel(Number(d.fused_conf || 0)).toLowerCase()}.`;
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
    const allEmotions = ['neutral', 'happy', 'sad', 'anger'];
    replayChart.data.datasets[0].data = allEmotions.map(e => scores[e] || 0);
    replayChart.data.datasets[0].backgroundColor = allEmotions.map(e =>
        e === highlight ? (EMOTION_COLORS[e] || '#666') : '#333'
    );
    replayChart.update();
}
