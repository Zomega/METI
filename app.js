/**
 * ARECIBO AUDIO DECODER
 * Core Logic: Signal Parsing, FSK Synthesis, and Canvas Rendering.
 */

class AreciboDecoder {
    constructor() {
        this.bits = [];
        this.currentIndex = 0;
        this.isPlaying = false;
        this.playbackInterval = null;
        this.bitRate = 200; // Slower default (5 bits per second)
        this.gridWidth = 23;

        // Audio setup
        this.audioCtx = null;
        this.oscillator = null;
        this.gainNode = null;
        this.noiseNode = null;
        this.noiseGain = null;
        this.freq0 = 1200; // Researched low tone
        this.freq1 = 2200; // Researched high tone

        // DOM elements
        this.canvas = document.getElementById('decoder-grid');
        this.ctx = this.canvas.getContext('2d');
        this.widthSlider = document.getElementById('width-slider');
        this.widthValue = document.getElementById('width-value');
        this.speedSlider = document.getElementById('speed-slider');
        this.speedValue = document.getElementById('speed-value');
        this.playBtn = document.getElementById('play-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.bitCounter = document.getElementById('bit-counter');
        this.bitIndicator = document.getElementById('bit-indicator');
        this.semiprimeAlert = document.getElementById('semiprime-alert');

        this.init();
    }

    async init() {
        await this.loadData();
        this.setupEventListeners();
        this.resizeCanvas();
        this.draw();
        
        // Set initial slider values
        this.speedSlider.value = this.bitRate;
        this.speedValue.innerText = this.bitRate;
    }

    async loadData() {
        try {
            const response = await fetch('Arecibo/message.txt');
            const text = await response.text();
            this.bits = text.replace(/\s+/g, '').split('').map(b => parseInt(b));
            console.log(`Loaded ${this.bits.length} bits.`);
        } catch (err) {
            console.error('Failed to load signal data:', err);
        }
    }

    setupEventListeners() {
        this.widthSlider.oninput = (e) => {
            this.gridWidth = parseInt(e.target.value);
            this.widthValue.innerText = this.gridWidth;
            this.checkSemiprime();
            this.draw();
        };

        this.speedSlider.oninput = (e) => {
            this.bitRate = parseInt(e.target.value);
            this.speedValue.innerText = this.bitRate;
            if (this.isPlaying) {
                this.stopPlayback();
                this.startPlayback();
            }
        };

        this.playBtn.onclick = () => {
            if (this.isPlaying) {
                this.stopPlayback();
            } else {
                this.startPlayback();
            }
        };

        this.stopBtn.onclick = () => {
            this.stopPlayback();
            this.currentIndex = 0;
            this.updateUI();
            this.draw();
        };

        window.onresize = () => this.resizeCanvas();
    }

    resizeCanvas() {
        // Match CSS width, but calculate height based on rows
        const padding = 2;
        const cellSize = (this.canvas.offsetWidth - padding) / this.gridWidth;
        const rows = Math.ceil(this.bits.length / this.gridWidth);
        
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = rows * cellSize;
        this.draw();
    }

    initAudio() {
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            
            // Signal path
            this.gainNode = this.audioCtx.createGain();
            this.gainNode.connect(this.audioCtx.destination);
            this.gainNode.gain.value = 0; // Start muted

            // Background noise path
            this.createWhiteNoise();
        }
    }

    createWhiteNoise() {
        const bufferSize = 2 * this.audioCtx.sampleRate;
        const noiseBuffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }

        this.noiseNode = this.audioCtx.createBufferSource();
        this.noiseNode.buffer = noiseBuffer;
        this.noiseNode.loop = true;

        this.noiseGain = this.audioCtx.createGain();
        this.noiseGain.gain.value = 0; // Start muted
        
        this.noiseNode.connect(this.noiseGain);
        this.noiseGain.connect(this.audioCtx.destination);
        this.noiseNode.start();
    }

    startPlayback() {
        this.initAudio();
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }

        this.isPlaying = true;
        this.playBtn.innerText = 'PAUSE_SIGNAL';

        const now = this.audioCtx.currentTime;
        
        // Start drone oscillator
        this.oscillator = this.audioCtx.createOscillator();
        this.oscillator.type = 'sine';
        this.oscillator.connect(this.gainNode);
        this.oscillator.start();

        // Fade in signal and noise
        this.gainNode.gain.setTargetAtTime(0.1, now, 0.05);
        this.noiseGain.gain.setTargetAtTime(0.02, now, 0.05);
        
        this.playbackInterval = setInterval(() => {
            if (this.currentIndex < this.bits.length) {
                this.updateSignal(this.bits[this.currentIndex]);
                this.updateUI();
                this.currentIndex++;
                this.draw(); 
            } else {
                this.stopPlayback();
            }
        }, this.bitRate);
    }

    stopPlayback() {
        if (!this.isPlaying) return;
        
        this.isPlaying = false;
        this.playBtn.innerText = 'RESUME_SIGNAL';
        clearInterval(this.playbackInterval);

        const now = this.audioCtx.currentTime;
        
        // Fade out signal and noise
        this.gainNode.gain.setTargetAtTime(0, now, 0.05);
        this.noiseGain.gain.setTargetAtTime(0, now, 0.05);

        // Stop oscillator after fade out
        setTimeout(() => {
            if (!this.isPlaying && this.oscillator) {
                this.oscillator.stop();
                this.oscillator = null;
            }
        }, 200);
    }

    updateSignal(bit) {
        if (!this.oscillator) return;
        
        const now = this.audioCtx.currentTime;
        // Instant frequency shift for the continuous drone
        this.oscillator.frequency.setTargetAtTime(bit === 1 ? this.freq1 : this.freq0, now, 0.005);

        // UI Feedback
        this.bitIndicator.className = bit === 1 ? 'active-1' : 'active-0';
    }

    updateUI() {
        this.bitCounter.innerText = `BIT: ${this.currentIndex} / ${this.bits.length}`;
    }

    checkSemiprime() {
        if (this.gridWidth === 23 || this.gridWidth === 73) {
            this.semiprimeAlert.classList.add('visible');
        } else {
            this.semiprimeAlert.classList.remove('visible');
        }
    }

    draw() {
        const { ctx, canvas, bits, gridWidth, currentIndex } = this;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (bits.length === 0) return;

        const padding = 2;
        const cellSize = (canvas.width - padding) / gridWidth;

        // Only draw up to currentIndex
        for (let i = 0; i < currentIndex; i++) {
            const x = (i % gridWidth) * cellSize;
            const y = Math.floor(i / gridWidth) * cellSize;

            if (bits[i] === 1) {
                ctx.fillStyle = '#00ff41';
                // Highlight the most recently added bit
                if (i === currentIndex - 1) {
                    ctx.shadowBlur = 15;
                    ctx.shadowColor = '#00ff41';
                } else {
                    ctx.shadowBlur = 0;
                }
                ctx.fillRect(x + 1, y + 1, cellSize - 1, cellSize - 1);
            } else {
                ctx.strokeStyle = '#1a1a1a';
                ctx.strokeRect(x + 1, y + 1, cellSize - 1, cellSize - 1);
            }
        }
        ctx.shadowBlur = 0;
    }
}

// Boot the app
window.onload = () => {
    new AreciboDecoder();
};
