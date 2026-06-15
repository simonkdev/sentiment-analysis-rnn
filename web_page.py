INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>RNN Sentiment Lab</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #0b1020;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-strong: rgba(15, 23, 42, 0.96);
            --border: rgba(148, 163, 184, 0.22);
            --muted: #94a3b8;
            --text: #e5e7eb;
            --accent: #8b5cf6;
            --accent-2: #22d3ee;
            --positive: #34d399;
            --negative: #fb7185;
            --warning: #fbbf24;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
        }

        * {
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 20%, rgba(139, 92, 246, 0.34), transparent 28rem),
                radial-gradient(circle at 85% 10%, rgba(34, 211, 238, 0.22), transparent 26rem),
                linear-gradient(135deg, #020617 0%, var(--bg) 52%, #111827 100%);
        }

        .shell {
            width: min(1120px, calc(100% - 32px));
            margin: 0 auto;
            padding: 40px 0;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 28px;
            align-items: stretch;
        }

        .card {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 28px;
            background: var(--panel);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .intro {
            padding: 34px;
        }

        .badge {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 8px 12px;
            border: 1px solid rgba(139, 92, 246, 0.42);
            border-radius: 999px;
            color: #ddd6fe;
            background: rgba(139, 92, 246, 0.14);
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        h1 {
            margin: 22px 0 14px;
            font-size: clamp(40px, 7vw, 76px);
            line-height: 0.92;
            letter-spacing: -0.07em;
        }

        .gradient-text {
            background: linear-gradient(90deg, #f8fafc, #a78bfa 48%, #22d3ee);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .lede {
            max-width: 640px;
            margin: 0;
            color: #cbd5e1;
            font-size: 18px;
            line-height: 1.65;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 28px;
        }

        .stat {
            padding: 16px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(2, 6, 23, 0.32);
        }

        .stat strong {
            display: block;
            font-size: 20px;
        }

        .stat span {
            color: var(--muted);
            font-size: 12px;
        }

        .workspace {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 360px;
            gap: 22px;
            margin-top: 24px;
        }

        .input-card,
        .result-card {
            padding: 24px;
        }

        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 14px;
        }

        .section-title h2 {
            margin: 0;
            font-size: 18px;
            letter-spacing: -0.02em;
        }

        .hint {
            color: var(--muted);
            font-size: 13px;
        }

        textarea {
            width: 100%;
            min-height: 240px;
            resize: vertical;
            padding: 18px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 20px;
            outline: none;
            color: var(--text);
            background: rgba(2, 6, 23, 0.56);
            font: inherit;
            line-height: 1.55;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }

        textarea:focus {
            border-color: rgba(34, 211, 238, 0.74);
            box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.12);
        }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            justify-content: space-between;
            margin-top: 16px;
        }

        button {
            border: 0;
            border-radius: 16px;
            cursor: pointer;
            font: inherit;
            font-weight: 800;
            transition: transform 0.16s ease, opacity 0.16s ease, border-color 0.16s ease;
        }

        button:active {
            transform: translateY(1px);
        }

        .primary {
            padding: 14px 18px;
            color: #020617;
            background: linear-gradient(135deg, var(--accent-2), #a78bfa);
        }

        .secondary,
        .sample {
            color: #cbd5e1;
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.68);
        }

        .secondary {
            padding: 14px 16px;
        }

        .sample {
            padding: 10px 12px;
            font-size: 13px;
        }

        .samples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .result-card {
            background: var(--panel-strong);
        }

        .result-empty,
        .result-ready {
            min-height: 340px;
        }

        .result-empty {
            display: grid;
            place-items: center;
            text-align: center;
            color: var(--muted);
        }

        .orb {
            width: 104px;
            height: 104px;
            margin: 0 auto 20px;
            border-radius: 999px;
            background: radial-gradient(circle at 35% 35%, #f8fafc, #22d3ee 28%, #7c3aed 66%, #111827);
            box-shadow: 0 0 44px rgba(34, 211, 238, 0.26);
        }

        .sentiment {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .sentiment.positive {
            color: #bbf7d0;
            background: rgba(52, 211, 153, 0.16);
        }

        .sentiment.negative {
            color: #fecdd3;
            background: rgba(251, 113, 133, 0.16);
        }

        .confidence {
            margin: 26px 0;
        }

        .meter {
            margin-top: 14px;
        }

        .meter-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            color: #cbd5e1;
            font-size: 13px;
            font-weight: 700;
        }

        .track {
            height: 12px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
        }

        .fill {
            height: 100%;
            width: 0;
            border-radius: inherit;
            transition: width 0.4s ease;
        }

        .fill.positive {
            background: linear-gradient(90deg, #10b981, #86efac);
        }

        .fill.negative {
            background: linear-gradient(90deg, #f43f5e, #fda4af);
        }

        .note {
            padding: 14px;
            border: 1px solid rgba(251, 191, 36, 0.22);
            border-radius: 16px;
            color: #fde68a;
            background: rgba(251, 191, 36, 0.08);
            font-size: 13px;
            line-height: 1.5;
        }

        .footer {
            margin-top: 22px;
            color: var(--muted);
            text-align: center;
            font-size: 13px;
        }

        .hidden {
            display: none;
        }

        @media (max-width: 900px) {
            .hero,
            .workspace {
                grid-template-columns: 1fr;
            }

            .stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div class="card intro">
                <div class="badge">RNN sentiment classifier</div>
                <h1><span class="gradient-text">Review sentiment,</span><br />decoded in seconds.</h1>
                <p class="lede">
                    Paste a movie review and inspect how this handcrafted recurrent neural network classifies the tone.
                    The interface is built like a compact ML demo: input on the left, prediction and score breakdown on the right.
                </p>
                <div class="stats">
                    <div class="stat"><strong>IMDB</strong><span>review domain</span></div>
                    <div class="stat"><strong>RNN</strong><span>custom NumPy model</span></div>
                    <div class="stat"><strong>2-way</strong><span>positive / negative</span></div>
                </div>
            </div>
            <div class="card result-card">
                <div id="emptyState" class="result-empty">
                    <div>
                        <div class="orb"></div>
                        <h2>Awaiting review text</h2>
                        <p>Run an example or paste your own review to see the prediction.</p>
                    </div>
                </div>
                <div id="resultState" class="result-ready hidden">
                    <div class="section-title">
                        <h2>Prediction</h2>
                        <span id="latency" class="hint"></span>
                    </div>
                    <div id="sentimentBadge" class="sentiment">—</div>
                    <div class="confidence">
                        <div class="meter">
                            <div class="meter-label"><span>Positive</span><span id="positiveScore">0%</span></div>
                            <div class="track"><div id="positiveFill" class="fill positive"></div></div>
                        </div>
                        <div class="meter">
                            <div class="meter-label"><span>Negative</span><span id="negativeScore">0%</span></div>
                            <div class="track"><div id="negativeFill" class="fill negative"></div></div>
                        </div>
                    </div>
                    <div class="note">
                        This is a small educational RNN, not a production-grade sentiment model. Treat the result as a model demo.
                    </div>
                </div>
            </div>
        </section>

        <section class="workspace">
            <div class="card input-card">
                <div class="section-title">
                    <h2>Analyze a review</h2>
                    <span id="charCount" class="hint">0 characters</span>
                </div>
                <textarea id="inputText" placeholder="Example: The movie starts slowly, but the final act is emotional, beautifully acted, and worth watching."></textarea>
                <div class="actions">
                    <div>
                        <button class="primary" onclick="processText()">Analyze sentiment</button>
                        <button class="secondary" onclick="clearText()">Clear</button>
                    </div>
                    <span id="status" class="hint">Ready</span>
                </div>
                <div class="samples">
                    <button class="sample" onclick="useSample('positive')">Positive sample</button>
                    <button class="sample" onclick="useSample('negative')">Negative sample</button>
                    <button class="sample" onclick="useSample('mixed')">Mixed sample</button>
                </div>
            </div>
            <aside class="card input-card">
                <h2>How to read it</h2>
                <p class="hint">
                    The model returns two class scores. The larger score becomes the label. Stronger separation means the model
                    is more decisive; close scores mean the text is ambiguous or the model is uncertain.
                </p>
                <p class="hint">
                    For best results, use full English movie-review style text rather than one-word prompts.
                </p>
            </aside>
        </section>
        <p class="footer">Built with Flask, NumPy, and a from-scratch recurrent neural network.</p>
    </main>

    <script>
        const samples = {
            positive: "A warm, beautifully paced film with excellent performances and a surprisingly emotional ending. I would absolutely recommend it.",
            negative: "The plot is dull, the dialogue feels forced, and the whole movie wastes a promising idea with weak acting.",
            mixed: "The visuals are impressive and the actors try hard, but the story is uneven and the ending feels rushed."
        };

        const input = document.getElementById("inputText");
        const charCount = document.getElementById("charCount");
        const statusText = document.getElementById("status");

        input.addEventListener("input", updateCount);
        updateCount();

        function updateCount() {
            charCount.textContent = `${input.value.length} characters`;
        }

        function useSample(type) {
            input.value = samples[type];
            updateCount();
            processText();
        }

        function clearText() {
            input.value = "";
            updateCount();
            statusText.textContent = "Ready";
            document.getElementById("emptyState").classList.remove("hidden");
            document.getElementById("resultState").classList.add("hidden");
        }

        async function processText() {
            const text = input.value.trim();
            if (!text) {
                statusText.textContent = "Enter text first";
                return;
            }

            const startedAt = performance.now();
            statusText.textContent = "Analyzing…";

            try {
                const response = await fetch("/api/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || "Request failed");
                }
                renderResult(data, Math.round(performance.now() - startedAt));
                statusText.textContent = "Analysis complete";
            } catch (error) {
                statusText.textContent = error.message;
            }
        }

        function renderResult(data, latencyMs) {
            const result = data.result;
            const scores = data.scores || {};
            const positive = Math.round((scores.positive || 0) * 100);
            const negative = Math.round((scores.negative || 0) * 100);
            const badge = document.getElementById("sentimentBadge");

            badge.textContent = result;
            badge.className = `sentiment ${result}`;
            document.getElementById("positiveScore").textContent = `${positive}%`;
            document.getElementById("negativeScore").textContent = `${negative}%`;
            document.getElementById("positiveFill").style.width = `${positive}%`;
            document.getElementById("negativeFill").style.width = `${negative}%`;
            document.getElementById("latency").textContent = `${latencyMs} ms`;
            document.getElementById("emptyState").classList.add("hidden");
            document.getElementById("resultState").classList.remove("hidden");
        }
    </script>
</body>
</html>
"""
