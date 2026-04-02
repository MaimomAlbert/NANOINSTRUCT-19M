import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NanoInstruct · Chat",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #0a0a0f;
    color: #e2e8f0;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid #1e1e3a !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }

/* ── Sidebar text ── */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #94a3b8 !important; font-size: 0.85rem !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Slider & inputs ── */
.stSlider > div > div > div { background: #7c3aed !important; }
.stSlider > div > div { background: #1e1e3a !important; }
.stTextInput input, .stTextArea textarea {
    background: #111127 !important;
    border: 1px solid #2d2d5e !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

/* ── Main area ── */
.main-wrap {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #0a0a0f;
}

/* ── Header ── */
.top-bar {
    padding: 1rem 2rem;
    border-bottom: 1px solid #1e1e3a;
    background: #0d0d18;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.top-bar .logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #7c3aed;
    letter-spacing: -0.5px;
}
.top-bar .logo span { color: #e2e8f0; }
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.model-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: #1e1e3a;
    color: #a78bfa;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid #2d2d5e;
}

/* ── Chat container ── */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    scrollbar-width: thin;
    scrollbar-color: #2d2d5e transparent;
}

/* ── Messages ── */
.msg-row {
    display: flex;
    gap: 12px;
    max-width: 820px;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user  { margin-left: auto; flex-direction: row-reverse; }
.msg-row.model { margin-right: auto; }

.avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}
.avatar.user  { background: linear-gradient(135deg, #7c3aed, #4f46e5); color: white; }
.avatar.model { background: #1e1e3a; color: #a78bfa; border: 1px solid #2d2d5e; }

.bubble {
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.65;
    max-width: 680px;
    word-wrap: break-word;
}
.bubble.user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-bottom-right-radius: 4px;
}
.bubble.model {
    background: #111127;
    color: #e2e8f0;
    border: 1px solid #1e1e3a;
    border-bottom-left-radius: 4px;
    font-family: 'Outfit', sans-serif;
}
.bubble.model pre, .bubble.model code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: #0a0a0f;
    padding: 2px 6px;
    border-radius: 4px;
    color: #a78bfa;
}

/* ── Typing indicator ── */
.typing {
    display: flex; gap: 5px; align-items: center;
    padding: 0.85rem 1.1rem;
    background: #111127;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    width: fit-content;
}
.typing span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #7c3aed;
    animation: bounce 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30%            { transform: translateY(-6px); opacity: 1; }
}

/* ── Input bar ── */
.input-bar {
    padding: 1rem 2rem 1.5rem;
    border-top: 1px solid #1e1e3a;
    background: #0d0d18;
}

/* ── Welcome screen ── */
.welcome {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    text-align: center;
    padding: 3rem;
}
.welcome h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.welcome p { color: #64748b; font-size: 1rem; max-width: 420px; line-height: 1.6; }
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.5rem;
    max-width: 560px;
    width: 100%;
}
.suggestion {
    background: #111127;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
    font-size: 0.85rem;
    color: #94a3b8;
}
.suggestion:hover {
    border-color: #7c3aed;
    color: #e2e8f0;
    background: #1a1a35;
}
.suggestion strong { display: block; color: #c4b5fd; font-size: 0.78rem; margin-bottom: 3px; }

/* ── Stat cards ── */
.stat-card {
    background: #111127;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.stat-card .label { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; }
.stat-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #a78bfa; font-weight: 600; }

/* ── Clear button ── */
.stButton.clear > button {
    background: #1e1e3a !important;
    color: #ef4444 !important;
    border: 1px solid #3f1a1a !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model Architecture (must match training) ──────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.qkv        = nn.Linear(dim, dim * 3, bias=False)
        self.out        = nn.Linear(dim, dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv     = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def split(t): return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        cos, sin = self.rotary_emb(S, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.fc   = nn.Linear(dim, hidden_dim, bias=False)
        self.out  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.fc(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, max_seq_len):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn      = Attention(dim, num_heads, max_seq_len)
        self.ffn_norm  = nn.LayerNorm(dim)
        self.ffn       = FeedForward(dim, hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SimpleSLM(nn.Module):
    def __init__(self, vocab_size=256, dim=396, num_heads=6, num_layers=8,
                 hidden_dim=1536, max_seq_len=105):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb   = nn.Embedding(vocab_size, dim)
        self.layers      = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.token_emb.weight = self.head.weight

    def forward(self, x, mask=None, labels=None):
        S = x.shape[1]
        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=x.device))
        x      = self.token_emb(x)
        for layer in self.layers:
            x  = layer(x, mask)
        x      = self.norm(x)
        logits = self.head(x)
        loss   = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100,
            )
        return {'logits': logits, 'loss': loss} if loss is not None else logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=150, temperature=0.7, eos_token_id=3):
        generated = []
        for _ in range(max_new_tokens):
            logits   = self(input_ids[:, -self.max_seq_len:])[:, -1, :] / max(temperature, 1e-8)
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok.item())
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if next_tok.item() == eos_token_id:
                break
        return generated


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self):
        self.vocab_size   = 256
        self.pad_token_id = 2
        self.eos_token_id = 3

    def encode_prompt(self, text, max_length=104):
        ids = list(text.encode('utf-8')[:max_length])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.pad_token_id, self.eos_token_id]]
        return bytes(ids).decode('utf-8', errors='ignore')


# ── Load model (cached) ───────────────────────────────────────────────────────

MODEL_DIR = "nano_instruct_model"

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        with open(f"{MODEL_DIR}/config.json") as f:
            cfg = json.load(f)
        model = SimpleSLM(**cfg).to(device)
        model.load_state_dict(
            torch.load(f"{MODEL_DIR}/model.pt", map_location=device)
        )
        model.eval()
        return model, device, cfg, None
    except FileNotFoundError:
        return None, device, {}, f"Model not found at '{MODEL_DIR}/'. Please check the path."
    except Exception as e:
        return None, torch.device('cpu'), {}, str(e)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_turns" not in st.session_state:
    st.session_state.total_turns = 0


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading NanoInstruct model..."):
    model, device, cfg, load_error = load_model()

tokenizer = SimpleTokenizer()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ NanoInstruct")
    st.markdown("---")

    # Model info
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        st.markdown(f"""
<div class="stat-card">
  <div class="label">Parameters</div>
  <div class="value">{total_params/1e6:.1f}M</div>
</div>
<div class="stat-card">
  <div class="label">Architecture</div>
  <div class="value">{cfg.get('num_layers',8)}L · {cfg.get('num_heads',6)}H · {cfg.get('dim',396)}D</div>
</div>
<div class="stat-card">
  <div class="label">Device</div>
  <div class="value">{'🟢 CUDA' if str(device) != 'cpu' else '🔵 CPU'}</div>
</div>
<div class="stat-card">
  <div class="label">Turns · Tokens Out</div>
  <div class="value">{st.session_state.total_turns} · {st.session_state.total_tokens}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.error(f"❌ {load_error}")

    st.markdown("---")
    st.markdown("### 🎛️ Generation Settings")

    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05,
        help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max new tokens", 20, 200, 100, 10,
        help="Maximum tokens to generate")

    st.markdown("---")
    st.markdown("### 💬 Conversation")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_turns = 0
        st.session_state.total_tokens = 0
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.75rem; color:#334155; line-height:1.6;">
Built with PyTorch<br>
Trained on Alpaca dataset<br>
Byte-level tokenizer (vocab=256)<br>
RoPE + SwiGLU + Pre-Norm
</div>
""", unsafe_allow_html=True)


# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="status-dot"></div>
  <div class="logo">Nano<span>Instruct</span></div>
  <div class="model-badge">19.8M · SLM</div>
  <div class="model-badge">Alpaca Fine-tuned</div>
</div>
""", unsafe_allow_html=True)


# ── Generate response ─────────────────────────────────────────────────────────
def generate_response(instruction: str) -> str:
    if model is None:
        return f"❌ Model not loaded: {load_error}"
    prompt       = f"### Instruction:\n{instruction}\n\n### Response:\n"
    prompt_bytes = tokenizer.encode_prompt(prompt, max_length=104)
    input_ids    = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
    new_ids      = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    if not new_ids:
        return "*(no output — try a lower temperature or different prompt)*"
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    st.session_state.total_tokens += len(new_ids)
    return response or "*(empty response)*"


# ── Chat UI ───────────────────────────────────────────────────────────────────
def render_message(role: str, content: str):
    avatar = "U" if role == "user" else "N"
    cls    = "user" if role == "user" else "model"
    st.markdown(f"""
<div class="msg-row {cls}">
  <div class="avatar {cls}">{avatar}</div>
  <div class="bubble {cls}">{content}</div>
</div>
""", unsafe_allow_html=True)


# Welcome screen or chat history
if not st.session_state.messages:
    st.markdown("""
<div class="welcome">
  <h1>⚡ NanoInstruct</h1>
  <p>A 19.8M parameter instruction-tuned language model built from scratch.
     Ask me anything — I'll do my best!</p>
  <div class="suggestion-grid">
    <div class="suggestion"><strong>✍️ Writing</strong>Write a short poem about the ocean</div>
    <div class="suggestion"><strong>🧠 Explain</strong>Explain neural networks simply</div>
    <div class="suggestion"><strong>📋 List</strong>Give me 5 tips for better coding</div>
    <div class="suggestion"><strong>🌍 Knowledge</strong>What is machine learning?</div>
  </div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-bar">', unsafe_allow_html=True)
col_input, col_btn = st.columns([6, 1])

with col_input:
    user_input = st.text_input(
        "message",
        placeholder="Type an instruction or question…",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Handle send ───────────────────────────────────────────────────────────────
if (send or user_input) and user_input.strip():
    prompt = user_input.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_turns += 1

    # Generate
    with st.spinner(""):
        response = generate_response(prompt)

    # Add model message
    st.session_state.messages.append({"role": "model", "content": response})
    st.rerun()