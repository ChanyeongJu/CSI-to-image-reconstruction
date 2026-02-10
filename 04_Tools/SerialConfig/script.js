let port;
let reader;
let writer;
let inputBuffer = "";
let syncInterval = null;
let ackResolver = null;
let keepReading = true;

const btnConnect = document.getElementById('btnConnect');
const btnApply = document.getElementById('btnApply');
const btnRestart = document.getElementById('btnRestart');
const btnGetConfig = document.getElementById('btnGetConfig');
const btnClear = document.getElementById('btnClear');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const consoleDiv = document.getElementById('console');
const ipInput = document.getElementById('ipInput');
const portInput = document.getElementById('portInput');
const ssidInput = document.getElementById('ssidInput');
const pwdInput = document.getElementById('pwdInput');

const actionButtons = [ssidInput, pwdInput, ipInput, portInput, btnApply, btnRestart, btnGetConfig, btnClear];

if (!("serial" in navigator)) {
    statusText.textContent = "Web Serial API not supported";
    statusDot.style.backgroundColor = "var(--error)";
    btnConnect.disabled = true;
}

const log = (msg, type = '') => {
    const div = document.createElement('div');
    div.className = `log-entry ${type}`;
    div.textContent = `> ${msg}`;
    consoleDiv.appendChild(div);
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
};

const updateUI = (state) => {
    const states = {
        CONNECTED: { dot: 'connected', text: "Connected (115200bps)", btn: "Disconnect", disable: true },
        SYNCING: { dot: 'connected', text: "Syncing configuration...", btn: "Disconnect", disable: true },
        READY: { dot: 'connected', text: "Device Ready", btn: "Disconnect", disable: false },
        DISCONNECTED: { dot: '', text: "Device Not Connected", btn: "Connect Device", disable: true }
    };

    const s = states[state] || states.DISCONNECTED;
    statusDot.className = `status-dot ${s.dot}`;
    statusText.textContent = s.text;
    btnConnect.textContent = s.btn;
    actionButtons.forEach(btn => btn.disabled = s.disable);
};

async function readLoop() {
    keepReading = true;
    while (port?.readable && keepReading) {
        reader = port.readable.getReader();
        try {
            while (keepReading) {
                const { value, done } = await reader.read();
                if (done || !keepReading) break;

                inputBuffer += new TextDecoder().decode(value);
                if (inputBuffer.includes('\n')) {
                    const lines = inputBuffer.split('\n');
                    inputBuffer = lines.pop();
                    lines.forEach(handleLine);
                }
            }
        } catch (error) {
            if (keepReading) log("Read error: " + error, 'log-err');
        } finally {
            reader.releaseLock();
        }
    }
}

const handleLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    log(trimmed);

    if (ackResolver && (trimmed.startsWith("[OK]") || trimmed.startsWith("[ERR]"))) {
        ackResolver(trimmed);
        ackResolver = null;
    }

    if (trimmed.includes("[INFO] Current Config")) {
        const matches = {
            ssid: trimmed.match(/SSID:([^,]+)/),
            pwd: trimmed.match(/PWD:([^,]+)/),
            ip: trimmed.match(/IP:([^,]+)/),
            port: trimmed.match(/Port:(\d+)/)
        };

        if (matches.ssid) ssidInput.value = matches.ssid[1].trim();
        if (matches.pwd) pwdInput.value = matches.pwd[1].trim();
        if (matches.ip) ipInput.value = matches.ip[1].trim();
        if (matches.port) portInput.value = matches.port[1].trim();

        log("Device configuration synchronized.", "log-sys");
        stopSync();
        updateUI('READY');
    }

    if (trimmed.includes("[SYSTEM]") && trimmed.includes("Ready")) {
        log("Device reboot detected. Re-syncing...", "log-sys");
        startSync();
    }
};

const sendCmd = async (cmd) => {
    if (!writer) return;
    try {
        await writer.write(new TextEncoder().encode(cmd + "\n"));
        log(`Command sent: ${cmd}`, 'log-sys');
    } catch (e) {
        log(`Failed: ${e.message}`, 'log-err');
    }
};

const sendCmdWithAck = async (cmd, timeout = 2000) => {
    if (!writer) return { success: false, msg: "Not connected" };

    const ackPromise = new Promise(resolve => {
        ackResolver = resolve;
        setTimeout(() => {
            if (ackResolver) {
                ackResolver = null;
                resolve("[TIMEOUT] No response");
            }
        }, timeout);
    });

    await sendCmd(cmd);
    const response = await ackPromise;
    return { success: response.startsWith("[OK]"), msg: response };
};

const startSync = () => {
    stopSync();
    updateUI('SYNCING');
    sendCmd("GET_CONFIG");
    syncInterval = setInterval(() => port ? sendCmd("GET_CONFIG") : stopSync(), 1500);
};

const stopSync = () => {
    if (syncInterval) clearInterval(syncInterval);
    syncInterval = null;
};

const disconnect = async () => {
    keepReading = false;
    stopSync();

    if (writer) {
        try { writer.releaseLock(); } catch { }
        writer = null;
    }

    if (reader) {
        try { await reader.cancel(); } catch { }
        reader = null;
    }

    await new Promise(r => setTimeout(r, 100));
    try { await port?.close(); } catch { }
    port = null;

    updateUI('DISCONNECTED');
    log("Disconnected.", 'log-sys');
};

btnConnect.addEventListener('click', async () => {
    if (port) return disconnect();

    try {
        port = await navigator.serial.requestPort();
        await port.open({ baudRate: 115200 });
        writer = port.writable.getWriter();
        log("Connected.", 'log-sys');
        readLoop();
        startSync();
    } catch (err) {
        log("Failed: " + err.message, 'log-err');
        updateUI('DISCONNECTED');
    }
});

btnApply.addEventListener('click', async () => {
    const configs = [
        { cmd: 'SET_SSID', val: ssidInput.value, label: 'SSID' },
        { cmd: 'SET_PWD', val: pwdInput.value, label: 'Password' },
        { cmd: 'SET_IP', val: ipInput.value.trim(), label: 'IP' },
        { cmd: 'SET_PORT', val: portInput.value.trim(), label: 'Port' }
    ];

    updateUI('SYNCING');
    statusText.textContent = "Applying...";

    let errors = 0;
    for (const c of configs) {
        if (!c.val) continue;
        log(`Setting ${c.label}...`, "log-sys");
        const res = await sendCmdWithAck(`${c.cmd}:${c.val}`);
        res.success ? log(`✓ ${res.msg}`, "log-sys") : (log(`✗ ${res.msg}`, "log-err"), errors++);
        await new Promise(r => setTimeout(r, 50));
    }

    errors === 0 ? log("Success. Please reboot.", 'log-sys') : log(`Finished with ${errors} errors.`, 'log-err');
    updateUI('READY');
});

btnRestart.addEventListener('click', () => sendCmd("RESTART"));
btnGetConfig.addEventListener('click', () => sendCmd("GET_CONFIG"));
btnClear.addEventListener('click', () => {
    consoleDiv.innerHTML = '';
    log("Logs cleared.", 'log-sys');
});
