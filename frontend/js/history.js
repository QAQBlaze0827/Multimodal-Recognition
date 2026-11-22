// History uses the same server-side validity rules as analytics.
let historyRequest = 0;
let historyDetailRequest = 0;
let historySelected = null;

function historyText(zh, en) { return currentLang === 'zh' ? zh : en; }
function historyElement(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
}
function historyTime(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '--' : date.toLocaleString(currentLang === 'zh' ? 'zh-TW' : 'en-GB', { hour12: false });
}
function historyTop(item) {
    const entries = Object.entries(item.distribution);
    const maximum = Math.max(0, ...entries.map(([, count]) => count));
    if (!maximum) return historyText('沒有可參考的辨識結果', 'No usable recognition results');
    const labels = entries.filter(([, count]) => count === maximum).map(([emotion]) => formatEmotion(emotion)).join(' / ');
    const tied = entries.filter(([, count]) => count === maximum).length > 1;
    const pct = Math.round(maximum / item.valid_logs * 100);
    return tied
        ? historyText(`並列最多：${labels}（各 ${pct}%）`, `Tied most frequent: ${labels} (${pct}% each)`)
        : historyText(`最常辨識為${labels}（${pct}%）`, `Most often recognized as ${labels} (${pct}%)`);
}
function updateHistoryLanguage() {
    document.getElementById('history-period-label').textContent = tr('period');
    [...document.getElementById('history-days').options].forEach((option, index) => {
        option.textContent = tr(['sevenDays', 'thirtyDays', 'ninetyDays', 'year'][index]);
    });
    document.getElementById('history-disclaimer').textContent = historyText(
        '僅計選定期間內的紀錄；每次使用不代表同一位使用者。占比是紀錄比例，不是心情持續時間；模型分數不是準確率。',
        'Only records within the selected period are counted. Sessions may involve different people. Percentages describe records, not time spent feeling an emotion; model scores are not accuracy.');
}
function historyPager(id, total, offset, size, onPage) {
    const container = document.getElementById(id);
    container.replaceChildren();
    if (!total) return;
    const page = Math.floor(offset / size) + 1;
    for (const [symbol, label, next, disabled] of [
        ['\u2190', historyText('上一頁', 'Previous page'), offset - size, offset === 0],
        ['\u2192', historyText('下一頁', 'Next page'), offset + size, offset + size >= total],
    ]) {
        const button = historyElement('button', symbol);
        button.type = 'button';
        button.title = label;
        button.setAttribute('aria-label', label);
        button.disabled = disabled;
        button.addEventListener('click', () => onPage(next));
        container.append(button);
    }
    container.append(historyElement('span', historyText(`第 ${page} / ${Math.ceil(total / size)} 頁`, `Page ${page} / ${Math.ceil(total / size)}`)));
}
function initHistoryView() {
    document.getElementById('history-days').addEventListener('change', () => loadHistory());
    document.getElementById('history-refresh').addEventListener('click', () => loadHistory());
}
async function loadHistory(offset = 0) {
    const request = ++historyRequest;
    ++historyDetailRequest;
    historySelected = null;
    document.getElementById('history-detail').hidden = true;
    document.getElementById('history-sessions').replaceChildren();
    document.getElementById('history-pagination').replaceChildren();
    const status = document.getElementById('history-status');
    status.textContent = tr('loading');
    updateHistoryLanguage();
    try {
        const data = await apiGet('/api/history/sessions', { days: document.getElementById('history-days').value, limit: 10, offset });
        if (request !== historyRequest) return;
        if (!Number.isInteger(data.total) || !Array.isArray(data.items)) throw new Error('Invalid history response');
        if (offset && !data.items.length) return loadHistory(0);
        status.textContent = data.total
            ? historyText(`此期間有 ${data.total} 次使用紀錄`, `${data.total} sessions in this period`)
            : historyText('此期間尚無紀錄。', 'No records in this period.');
        const container = document.getElementById('history-sessions');
        data.items.forEach(item => {
            const row = historyElement('article', undefined, 'history-session');
            const text = historyElement('div');
            text.append(historyElement('h3', historyTime(item.first_record)));
            text.append(historyElement('p', historyText(`最後紀錄：${historyTime(item.last_record)}`, `Last record: ${historyTime(item.last_record)}`), 'summary-line'));
            text.append(historyElement('p', historyTop(item), 'history-top'));
            const coverage = Math.round(item.valid_logs / item.total_logs * 100);
            text.append(historyElement('p', historyText(
                `可參考 ${item.valid_logs} / ${item.total_logs} 筆（${coverage}%）；其中 ${item.low_score_logs} 筆模型分數低於 50%。`,
                `${item.valid_logs} / ${item.total_logs} usable records (${coverage}%); ${item.low_score_logs} of these scored below 50%.`)));
            text.append(historyElement('p', historyText(
                `未納入：無有效來源 ${item.no_source_logs} 筆、綜合結果無效 ${item.unresolved_logs} 筆。`,
                `Excluded: ${item.no_source_logs} without a valid source; ${item.unresolved_logs} without a valid combined result.`), 'summary-line'));
            const button = historyElement('button', historyText('查看明細', 'View details'));
            button.type = 'button';
            button.setAttribute('aria-label', historyText(`查看 ${historyTime(item.first_record)} 明細`, `Details for ${historyTime(item.first_record)}`));
            button.addEventListener('click', () => { historySelected = item; loadHistoryDetail(0, true); });
            row.append(text, button);
            container.append(row);
        });
        historyPager('history-pagination', data.total, offset, 10, loadHistory);
    } catch (error) {
        if (request !== historyRequest) return;
        containerClearHistory();
        status.textContent = historyText('紀錄載入失敗，請確認後端已更新後重新整理。', 'Unable to load history. Check the backend version and refresh.');
    }
}
function containerClearHistory() {
    document.getElementById('history-sessions').replaceChildren();
    document.getElementById('history-pagination').replaceChildren();
}
async function loadHistoryDetail(offset = 0, focus = false) {
    if (!historySelected) return;
    const item = historySelected;
    const request = ++historyDetailRequest;
    const detail = document.getElementById('history-detail');
    detail.hidden = false;
    const title = document.getElementById('history-detail-title');
    title.textContent = historyText(`辨識明細 · ${historyTime(item.first_record)}`, `Record details · ${historyTime(item.first_record)}`);
    if (focus) title.focus();
    const status = document.getElementById('history-detail-status');
    status.textContent = tr('loading');
    const body = document.querySelector('#history-records tbody');
    body.replaceChildren();
    document.getElementById('history-record-pagination').replaceChildren();
    const head = document.getElementById('history-record-head');
    head.replaceChildren(...historyText(['時間', '辨識結果', '模型分數', '紀錄狀態'], ['Time', 'Result', 'Model score', 'Record status']).map(label => historyElement('th', label)));
    try {
        const data = await apiGet('/api/history/records', { session_id: item.session_id, days: document.getElementById('history-days').value, limit: 20, offset });
        if (request !== historyDetailRequest) return;
        if (!Number.isInteger(data.total) || !Array.isArray(data.items)) throw new Error('Invalid records response');
        status.textContent = historyText(`共 ${data.total} 筆；無效紀錄不列入情緒占比。`, `${data.total} records; excluded records do not contribute to emotion percentages.`);
        for (const log of data.items) {
            const valid = log.quality === 'valid';
            const state = log.quality === 'no_source' ? historyText('無有效來源', 'No valid source')
                : !valid ? historyText('綜合結果無效', 'Invalid combined result')
                : log.fused_conf < 0.5 ? historyText('可參考 · 低分數', 'Usable · Low score') : historyText('可參考', 'Usable');
            const row = historyElement('tr');
            [historyTime(log.timestamp), valid ? formatEmotion(log.fused_emotion) : '--', valid ? `${Math.round(log.fused_conf * 100)}%` : '--', state]
                .forEach(value => row.append(historyElement('td', value)));
            body.append(row);
        }
        historyPager('history-record-pagination', data.total, offset, 20, loadHistoryDetail);
    } catch (error) {
        if (request !== historyDetailRequest) return;
        body.replaceChildren();
        status.textContent = historyText('明細載入失敗，請重新點選查看明細。', 'Details could not be loaded. Select View details to retry.');
    }
}
