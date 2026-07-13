const EMOTION_COLORS = {
    neutral:  '#9ca3af',
    happy:    '#22c55e',
    sad:      '#3b82f6',
    anger:    '#ef4444',
    fear:     '#a855f7',
    surprise: '#eab308',
    disgust:  '#f97316',
};

const EMOTION_LABELS = {
    zh: {
        neutral: '中性',
        happy: '開心',
        sad: '悲傷',
        anger: '生氣',
        fear: '害怕',
        surprise: '驚訝',
        disgust: '厭惡',
    },
    en: {
        neutral: 'Neutral',
        happy: 'Happy',
        sad: 'Sad',
        anger: 'Anger',
        fear: 'Fear',
        surprise: 'Surprise',
        disgust: 'Disgust',
    },
};

let pieChart = null;
let timelineChart = null;
let replayChart = null;

function chartLanguage() {
    return window.APP_LANG === 'en' ? 'en' : 'zh';
}

function chartEmotionLabel(emotion) {
    return EMOTION_LABELS[chartLanguage()][emotion] || emotion;
}

function createPieChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    if (pieChart) pieChart.destroy();
    const labels = Object.keys(data);
    const values = labels.map(l => data[l].count);
    const colors = labels.map(l => EMOTION_COLORS[l] || '#666');
    pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels.map(chartEmotionLabel),
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: '#1a1d27',
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#e8eaed', padding: 12, font: { size: 12 } },
                },
            },
        },
    });
    return pieChart;
}

function createTimelineChart(canvasId, timelineData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    if (timelineChart) timelineChart.destroy();

    const labels = timelineData.map(d => d.timestamp);
    const emotions = [...new Set(timelineData.map(d => d.fused_emotion))];

    const datasets = emotions.map(emotion => ({
        label: chartEmotionLabel(emotion),
        data: timelineData.map(d => d.fused_emotion === emotion ? d.fused_conf : 0),
        backgroundColor: EMOTION_COLORS[emotion] || '#666',
        borderColor: EMOTION_COLORS[emotion] || '#666',
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        tension: 0.2,
    }));

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    ticks: { color: '#9aa0a6', maxTicksLimit: 10, font: { size: 10 } },
                    grid: { color: '#2d3148' },
                },
                y: {
                    min: 0, max: 1,
                    ticks: { color: '#9aa0a6', font: { size: 10 } },
                    grid: { color: '#2d3148' },
                },
            },
            plugins: {
                legend: { labels: { color: '#e8eaed', font: { size: 11 } } },
            },
        },
    });
    return timelineChart;
}

function createReplayChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    if (replayChart) replayChart.destroy();
    const emotions = ['neutral', 'happy', 'sad', 'anger', 'fear', 'surprise', 'disgust'];
    replayChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: emotions.map(chartEmotionLabel),
            datasets: [{
                label: chartLanguage() === 'zh' ? '可信度' : 'Confidence',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: emotions.map(emotion => EMOTION_COLORS[emotion]),
                borderColor: '#1a1d27',
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            indexAxis: 'y',
            scales: {
                x: { min: 0, max: 1, ticks: { color: '#9aa0a6', font: { size: 10 } }, grid: { color: '#2d3148' } },
                y: { ticks: { color: '#e8eaed', font: { size: 11 } }, grid: { display: false } },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
    return replayChart;
}
