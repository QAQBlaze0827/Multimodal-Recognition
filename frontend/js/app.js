let currentView = 'live';
let replayData = [];
let replayIndex = 0;
let replayTimer = null;
let replaySpeed = 1;
let analyticsRequestId = 0;
let liveTrend = [];
let lastLiveData = null;
let lastLiveReceivedAt = null;
const LIVE_STALE_MS = 5000;
let companionHappyCount = 0;
let companionActionId = 0;
const TREND_LIMIT = 12;
let currentLang = localStorage.getItem('uiLang') || 'zh';
window.APP_LANG = currentLang;

const UI_TEXT = {
    zh: {
        companion: '互動陪伴', companionTitle: '柴犬夥伴', companionTouch: '摸摸柴犬',
        companionIdle: '安靜陪伴中', companionNoFace: '等待有效臉部訊號',
        companionTilt: '被摸摸了，歪頭回應', companionJump: '收到連續的開心表情，輕跳回應',
        appTitle: '多模態情緒辨識儀表板',
        pageTitle: '多模態情緒辨識儀表板',
        wsLabel: '後端連線：',
        dataWaiting: '等待辨識資料',
        dataWaitingNote: '尚未收到辨識紀錄',
        dataFresh: '資料持續更新',
        dataFreshNote: '已收到近期辨識紀錄',
        dataStale: '資料已停止更新',
        dataStaleNote: '顯示最後一次結果，非即時判斷。',
        dataDisconnected: '無法確認資料更新',
        dataDisconnectedNote: '後端連線尚未恢復；下方保留最後一次結果。',
        dataUnknown: '資料時間無法確認',
        dataUnknownNote: '紀錄時間缺失、無效或超前；下方結果不視為即時資料。',
        lastReading: '最後一次判讀',
        lastDataTime: '最後資料時間',
        secondsAgo: '秒前',
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
        analyticsTitle: '綜合辨識統計',
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
        companion: 'Companion', companionTitle: 'Shiba companion', companionTouch: 'Pet the Shiba',
        companionIdle: 'Resting nearby', companionNoFace: 'Waiting for a face signal',
        companionTilt: 'Tilting its head at your touch', companionJump: 'Hopping after a series of happy expressions',
        appTitle: 'Multimodal Emotion Dashboard',
        pageTitle: 'Multimodal Emotion Dashboard',
        wsLabel: 'Backend: ',
        dataWaiting: 'Waiting for recognition data',
        dataWaitingNote: 'No recognition records received',
        dataFresh: 'Data updating',
        dataFreshNote: 'Recent recognition record received',
        dataStale: 'Data updates stopped',
        dataStaleNote: 'Showing the last result, not a live reading.',
        dataDisconnected: 'Updates cannot be verified',
        dataDisconnectedNote: 'Backend connection unavailable; showing the last result.',
        dataUnknown: 'Record time unavailable',
        dataUnknownNote: 'Missing, invalid or future timestamp; results are not treated as live.',
        lastReading: 'Last Reading',
        lastDataTime: 'Last record time',
        secondsAgo: 'seconds ago',
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
        analyticsTitle: 'Fused Recognition Statistics',
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

    document.getElementById('companion-dog').addEventListener('click', () => playCompanion('tilt'));
    onWsMessage(data => acceptLiveData(data, true));
    setInterval(refreshLiveFreshness, 1000);
    document.addEventListener('visibilitychange', refreshLiveFreshness);
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
    setText('nav a[data-view="companion"]', 'companion');
    setText('#companion-title', 'companionTitle');
    setText('#companion-touch', 'companionTouch');
    document.getElementById('companion-dog').title = tr('companionTouch');
    if (typeof renderCompanionDialogue === 'function') renderCompanionDialogue();
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
    setText('#history-heading', 'historyTitle');
    setText('#analytics-heading', 'analyticsTitle');
    setText('#insight-face', 'waiting');
    setText('#insight-audio', 'waiting');
    setText('#insight-agreement', 'waiting');
    setText('#replay-load', 'load');
    setText('#replay-play', replayTimer ? 'pause' : 'play');
    setText('#history-refresh', 'refresh');
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
    if (typeof updateHistoryLanguage === 'function') updateHistoryLanguage();
    updateReplayStaticLabels();
    updateStatusLanguage();
    if (lastLiveData) updateLiveView(lastLiveData, false);
    else refreshLiveFreshness();
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
    resetCompanion();
    currentView = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');
    document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
    document.querySelector(`nav a[data-view="${view}"]`).classList.add('active');

    if (view === 'history') loadHistory();
    if (view === 'analytics') loadAnalytics();
    if (view === 'replay') loadReplaySessions();
    if (view === 'live' || view === 'companion') refreshLiveFreshness();
}

function initLiveView() {
    getRecentLogs(1).then(logs => {
        if (logs.length > 0) acceptLiveData(logs[0]);
    }).catch(() => {});
}

function acceptLiveData(data, fromStream = false) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) return;
    if (lastLiveData) {
        // Database IDs reject duplicate/reordered messages and late initial HTTP responses.
        if (Number.isInteger(data.id) && Number.isInteger(lastLiveData.id)) {
            if (data.id <= lastLiveData.id) return;
        } else {
            const previousTime = Date.parse(lastLiveData.timestamp);
            const nextTime = Date.parse(data.timestamp);
            if (data.timestamp === lastLiveData.timestamp
                    || (Number.isFinite(previousTime) && Number.isFinite(nextTime) && nextTime <= previousTime)) return;
        }
    }
    if (liveFreshness().state !== 'fresh' || lastLiveData?.session_id !== data.session_id ||
        Date.parse(data.timestamp) - Date.parse(lastLiveData?.timestamp) >= LIVE_STALE_MS) companionHappyCount = 0;
    lastLiveData = data;
    lastLiveReceivedAt = Date.now();
    updateLiveView(data);
    if (fromStream && currentView === 'companion' && !document.hidden && liveFreshness().state === 'fresh') {
        if (data.face_detected && data.video_conf > 0 && data.video_emotion === 'happy') {
            if (++companionHappyCount === 3) playCompanion('jump');
        } else {
            companionHappyCount = 0;
        }
    }
}

function resetCompanion() {
    companionHappyCount = 0;
    companionActionId++;
    document.getElementById('companion-dog').dataset.pose = 'idle';
}

function playCompanion(pose) {
    const dog = document.getElementById('companion-dog');
    const id = ++companionActionId;
    dog.dataset.pose = pose;
    renderCompanion(liveFreshness().state);
    setTimeout(() => {
        if (id !== companionActionId) return;
        dog.dataset.pose = 'idle';
        renderCompanion(liveFreshness().state);
    }, 800);
}

function renderCompanion(state) {
    if (currentView !== 'companion') return;
    const dog = document.getElementById('companion-dog');
    const validFace = state === 'fresh' && lastLiveData.face_detected && lastLiveData.video_conf > 0;
    if (!validFace || document.hidden) {
        companionHappyCount = 0;
        if (dog.dataset.pose === 'jump') resetCompanion();
    }
    const statusKey = dog.dataset.pose === 'tilt' ? 'companionTilt'
        : dog.dataset.pose === 'jump' ? 'companionJump'
        : validFace ? 'companionIdle'
        : ({ waiting: 'dataWaiting', disconnected: 'dataDisconnected', unknown: 'dataUnknown',
             stale: 'dataStale', fresh: 'companionNoFace' })[state];
    const status = document.getElementById('companion-status');
    if (status.textContent !== tr(statusKey)) status.textContent = tr(statusKey);
}

function liveFreshness(now = Date.now()) {
    if (!lastLiveData) return { state: 'waiting', age: null };
    const timestamp = typeof lastLiveData.timestamp === 'string' ? Date.parse(lastLiveData.timestamp) : NaN;
    const age = Number.isFinite(timestamp) && timestamp <= now + 1000
        ? Math.max(0, now - timestamp, now - lastLiveReceivedAt) : null;
    const connected = document.getElementById('ws-status').dataset.state === 'connected';
    if (!connected) return { state: 'disconnected', age };
    if (age === null) return { state: 'unknown', age };
    return { state: age >= LIVE_STALE_MS ? 'stale' : 'fresh', age };
}

function refreshLiveFreshness() {
    const { state, age } = liveFreshness();
    renderCompanion(state);
    const keys = {
        waiting: ['dataWaiting', 'dataWaitingNote'],
        fresh: ['dataFresh', 'dataFreshNote'],
        stale: ['dataStale', 'dataStaleNote'],
        disconnected: ['dataDisconnected', 'dataDisconnectedNote'],
        unknown: ['dataUnknown', 'dataUnknownNote'],
    };
    document.getElementById('live-freshness').dataset.state = state;
    const label = document.getElementById('live-data-label');
    const note = document.getElementById('live-data-note');
    // Avoid repeating live-region announcements on every timer tick.
    if (label.textContent !== tr(keys[state][0])) label.textContent = tr(keys[state][0]);
    if (note.textContent !== tr(keys[state][1])) note.textContent = tr(keys[state][1]);
    const lastUpdate = document.getElementById('live-last-update');
    lastUpdate.textContent = `${tr('lastDataTime')}：${age === null ? '--' : `${Math.floor(age / 1000)} ${tr('secondsAgo')}`}`;
    lastUpdate.title = lastLiveData && typeof lastLiveData.timestamp === 'string' ? lastLiveData.timestamp : '';
    const stale = Boolean(lastLiveData) && state !== 'fresh';
    document.getElementById('view-live').classList.toggle('is-stale', stale);
    document.querySelector('#view-live .live-summary h2').textContent = tr(stale ? 'lastReading' : 'currentReading');
    if (lastLiveData) {
        updateLiveSummary(lastLiveData);
        if (stale) document.getElementById('summary-line').textContent = tr(keys[state][1]);
    }
    const dot = document.querySelector('header .dot');
    if (dot) {
        dot.style.background = state === 'fresh' ? (EMOTION_COLORS[lastLiveData.fused_emotion] || 'var(--success)') : 'var(--text-muted)';
        dot.style.animationPlayState = state === 'fresh' ? 'running' : 'paused';
    }
}

function updateLiveView(data, appendTrend = true) {
    updateEmotionDisplay('fused', data.fused_emotion, data.fused_conf, tr('fusedSource'));
    updateEmotionDisplay('video', data.video_emotion || '-', data.video_conf || 0, tr('videoSource'));
    updateEmotionDisplay('audio', data.audio_emotion || '-', data.audio_conf || 0, tr('audioSource'));

    const scores = readScores(data, 'video');
    renderBars('live-bars', scores, data.fused_emotion);
    updateLiveSummary(data);
    updateSignalQuality(data);
    updateLiveTrend(data, appendTrend);

    document.getElementById('live-fps').textContent = data.fps ? data.fps.toFixed(1) : '-';
    document.getElementById('live-temp').textContent = data.cpu_temp ? `${data.cpu_temp.toFixed(1)}°C` : '-';
    document.getElementById('live-face').textContent = data.face_detected ? '有' : '無';

    refreshLiveFreshness();
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


function initAnalyticsView() {
    document.getElementById('analytics-days').addEventListener('change', loadAnalytics);
}

async function loadAnalytics() {
    const requestId = ++analyticsRequestId;
    const days = parseInt(document.getElementById('analytics-days').value) || 30;
    clearAnalytics();
    document.getElementById('analytics-status').textContent = tr('loading');
    try {
        const data = await getAnalytics(days);
        if (requestId !== analyticsRequestId) return;
        if (!Number.isInteger(data.valid_logs)) throw new Error('Analytics response needs updated backend');
        renderAnalytics(data);
    } catch (e) {
        if (requestId !== analyticsRequestId) return;
        clearAnalytics();
        document.getElementById('analytics-status').textContent = currentLang === 'zh'
            ? '統計載入失敗，請確認後端已更新並重新整理頁面。'
            : 'Statistics could not be loaded. Check the updated backend and reload.';
        console.error('Analytics load error:', e);
    }
}

function clearAnalytics() {
    if (pieChart) {
        pieChart.destroy();
        pieChart = null;
    }
    document.getElementById('analytics-distribution').hidden = true;
    for (const id of ['analytics-summary', 'analytics-table-head', 'analytics-table-body',
        'analytics-range', 'analytics-quality', 'analytics-insight', 'analytics-status']) {
        document.getElementById(id).textContent = '';
    }
}

function renderAnalytics(data) {
    const zh = currentLang === 'zh';
    const threshold = Math.round(data.low_score_threshold * 100);
    const metrics = [
        [data.total_logs, zh ? '總紀錄' : 'Total records'],
        [data.valid_logs, zh ? '有效紀錄（分布分母）' : 'Valid records (denominator)'],
        [data.low_score_logs, zh ? `其中低分數（低於 ${threshold}%）` : `Of valid: score below ${threshold}%`],
        [data.excluded_logs, zh ? '未納入分布' : 'Excluded records'],
    ];
    document.getElementById('analytics-summary').innerHTML = metrics.map(([count, label]) =>
        `<div class="stat-box"><div class="value">${Number(count)}</div><div class="label">${label}</div></div>`
    ).join('');
    document.getElementById('analytics-status').textContent = data.total_logs === 0
        ? (zh ? '選取期間沒有紀錄。' : 'No records in this period.')
        : data.valid_logs === 0
            ? (zh ? '有紀錄，但沒有可納入情緒分布的有效結果。' : 'Records exist, but none have a valid result for the distribution.')
            : '';
    document.getElementById('analytics-range').textContent = data.total_logs === 0 ? '' : (zh
        ? `實際紀錄範圍：${data.first_record} ～ ${data.last_record}；共 ${data.session_count} 次使用紀錄。`
        : `Recorded range: ${data.first_record} to ${data.last_record}; ${data.session_count} sessions.`);
    document.getElementById('analytics-quality').textContent = zh
        ? `未納入：無有效來源或來源欄位不足 ${data.no_source_logs} 筆；有來源但綜合結果無效 ${data.unresolved_logs} 筆。有效紀錄須有有效臉部或語音來源，且綜合類別與分數有效。`
        : `Excluded: ${data.no_source_logs} without a valid source or sufficient source fields; ${data.unresolved_logs} with a source but no valid fused result. Valid records require a valid face or audio source and a valid fused label and score.`;
    document.getElementById('analytics-insight').textContent = zh
        ? `情緒占比以 ${data.valid_logs} 筆有效紀錄為分母，包含低分數紀錄。${threshold}% 僅為分數分類門檻，不代表對錯；模型分數不是準確率，紀錄占比不是情緒持續時間或心理評估。`
        : `Distribution uses ${data.valid_logs} valid records, including low scores. ${threshold}% is a grouping threshold, not a correctness test. Model scores are not accuracy; record shares are not emotion duration or a psychological assessment.`;
    if (!data.valid_logs) return;

    document.getElementById('analytics-distribution').hidden = false;
    const headings = zh ? ['情緒', '筆數', '紀錄占比', '平均模型分數']
        : ['Emotion', 'Records', 'Record share', 'Mean model score'];
    document.getElementById('analytics-table-head').innerHTML = headings.map(text => `<th scope="col">${text}</th>`).join('');
    document.getElementById('analytics-table-body').innerHTML = ['neutral', 'happy', 'sad', 'anger'].map(emotion => {
        const item = data.distribution[emotion];
        return `<tr><td>${formatEmotion(emotion)}</td><td>${Number(item?.count || 0)}</td>
            <td>${Number(item?.pct || 0).toFixed(1)}%</td><td>${item ? (Number(item.avg_conf) * 100).toFixed(1) + '%' : '--'}</td></tr>`;
    }).join('');
    document.getElementById('analytics-pie').setAttribute('aria-label', zh ? '有效綜合情緒紀錄分布' : 'Valid fused emotion record distribution');
    if (typeof Chart === 'undefined') {
        document.getElementById('analytics-status').textContent = zh
            ? '圖表元件未載入，統計數值仍可查看。' : 'Chart library unavailable; statistics remain available in the table.';
    } else {
        createPieChart('analytics-pie', data.distribution);
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
