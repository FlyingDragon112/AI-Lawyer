# ⚖️ AI Lawyer — AI Legal Helper for Everyday Criminal-Law Problems in India

A conversational assistant that helps ordinary people in India understand where they stand when they face problems like **threats, cheating/fraud, theft, assault or harassment**. It turns a messy story into a structured case file, checks it against the **Bharatiya Nyaya Sanhita (BNS), 2023**, explains the relevant sections in plain language, and can draft a police complaint or let you rehearse the conversation with the other side.

> **Disclaimer:** This is an AI project, **not a lawyer** and **not legal advice**. It provides general information only. For anything serious or time-sensitive, please consult a qualified lawyer. Free legal aid is available via your District Legal Services Authority (NALSA helpline **15100**).

---

## ✨ Features

- **Case builder** – Converts your story into a structured case file: summary, other party, place, timeline, evidence (have / can get / don't have), harm suffered, steps already taken, and missing information. It is shown live in the sidebar.
- **Grounded legal explanation** – Cites only BNS sections that were retrieved from the database and judged relevant to your facts. It never invents law.
- **Document drafting** – Drafts a police complaint, a legal/demand notice (non-criminal matters), or a one-page case summary for a lawyer. Unknown details are left as `[PLACEHOLDERS]` instead of being made up.
- **Practice mode** – Plays the other side (for example, the police officer receiving your complaint) so you can rehearse, then gives you friendly, concrete feedback when you type `stop`.
- **Multilingual** – Understands and replies in English, Hindi and Hinglish (BGE-M3 is a multilingual embedder).
- **Safety routing** – Detects emergencies (shows 112 and other helplines), and routes people who have been accused, arrested or served a notice towards a real lawyer instead of DIY advice.
- **Honest about its limits** – For civil matters (deposits, salary, refunds, etc.) it still helps organise the case but cites **no** statute, because there is no verified source for it.

---

## 🧠 How It Works

The chatbot is a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine (`src/chatbot.py`) backed by a semantic search index of BNS sections (`src/retriever.py`).

```
User message
    │
    ▼
 triage ──► emergency ─────────────────────────────► END
    │  ├──► handoff (accused / notice received) ────► END
    │  └──► practice / practice_feedback ───────────► END
    ▼
 update_case  (structured case file)
    ▼
 retrieve  (exact lookup → semantic search → distance gate → LLM fit check → allow-list)
    ▼
 advise  or  draft
    ▼
 verify  (check every cited section; one rewrite, then redact) ──► END
```

### How it avoids invented law

| Layer | What it does |
|---|---|
| **Distance gate** | If the best retrieved section is too far from the query (cosine distance > `0.50`), no section is offered. |
| **LLM judge** | Every candidate section is judged `strong` / `possible` / `no` fit against the facts. Only the first two survive. |
| **Allow-list** | The writer may only cite sections that passed the judge. Section numbers are checked in code, not by the LLM. |
| **Verify node** | Regex-extracts every section number in the reply. Unknown ones trigger one rewrite, then are redacted. Case-law citations are blocked outright. |
| **Exact lookup** | If you name a section, it is checked against the index. A non-existent section is reported as missing. |
| **Escalation** | Emergency, accused or notice-received situations are routed away from "DIY" mode. |

---

## 🗂️ Project Structure

```
AI-Lawyer/
├── app.py                  # Streamlit front end
├── requirement.txt         # Python dependencies
├── Data Sources.md         # Dataset and model credits
├── Data/
│   └── bns_sections.csv    # BNS sections (358 rows)
└── src/
    ├── __init__.py
    ├── chatbot.py          # LangGraph conversation logic + citation guard
    └── retriever.py        # ChromaDB index + BGE-M3 semantic search
```

A `bns_chroma/` folder (the vector index) is created automatically on the first run and is git-ignored.

---

## 🛠️ Tech Stack

- **UI:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) + [LangChain](https://www.langchain.com/)
- **LLM:** Google Gemini via `langchain-google-genai` (default: `gemini-2.5-flash-lite`)
- **Embeddings:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via `FlagEmbedding`
- **Vector store:** [ChromaDB](https://www.trychroma.com/) (cosine similarity)
- **Data handling:** pandas, python-dotenv

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (the project was developed on Python 3.12)
- A **Google Gemini API key** (a free tier is available from [Google AI Studio](https://aistudio.google.com/))
- A few GB of free disk space and some patience on first run: BGE-M3 is downloaded and the BNS index is built (takes a few minutes; a GPU speeds this up but is not required)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/FlyingDragon112/AI-Lawyer.git
cd AI-Lawyer

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt
```

### Configuration

Create a `.env` file containing your API key. It can live in either the project root or in `src/`:

```env
GOOGLE_API_KEY=your-api-key-here
```

### Run the app

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually http://localhost:8501).

> **First run:** the BNS index is built automatically the first time a search is needed, so the first reply will be slow. You can also build it ahead of time:
> ```bash
> python -m src.retriever
> ```

---

## 💬 Example Prompts

- *"Someone keeps threatening to kill me on WhatsApp"*
- *"A person took ₹50,000 from me on a false promise and vanished"*
- *"My landlord is not returning my security deposit"* (civil matter → no statute is cited)
- *"Please draft the complaint I need, using what I've told you."*
- *"I want to practise the conversation with the other side."*

---

## 🐍 Using It as a Library

You can use the chatbot without the UI, from a script or a notebook (run from the project root):

```python
from src.chatbot import chat, show_case

print(chat("Someone keeps threatening to kill me over WhatsApp", thread_id="user1"))
print(chat("What can I do?", thread_id="user1"))
print(show_case("user1"))   # structured case file built so far
```

---

## ⚙️ Configuration Options

Settings live at the top of `src/chatbot.py`:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `google_genai:gemini-2.5-flash-lite` | Any LangChain chat-model string. Switch to a stronger model if replies or structured outputs are weak. |
| `TOP_K` | `5` | Number of BNS sections retrieved per search. |
| `NO_MATCH_DISTANCE` | `0.50` | Cosine-distance cutoff above which no section is offered. Tune on your own test queries. |
| `MAX_SOURCE_CHARS` | `1800` | Max characters of each section's text passed to the writer. |
| `HISTORY_MESSAGES` | `14` | How many recent messages are kept as context. |

---

## ⚠️ Limitations

- Covers **criminal law under the BNS only**, with a focus on offences against the person (threats, cheating/fraud, theft, assault, harassment).
- Does **not** map old IPC section numbers to BNS sections, and does not cite BNSS, IT Act, other statutes or case law.
- Conversation memory is in-process (`MemorySaver`); it is lost when the server restarts.
- LLM output can still be wrong or incomplete. Always verify with a qualified lawyer before acting.

---

## 📚 Data Sources & Credits

- **BNS dataset:** [Bharatiya Nyaya Sanhita Dataset (Kaggle)](https://www.kaggle.com/datasets/nandr39/bharatiya-nyaya-sanhita-dataset-bns)
- **Embedder / retriever model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

---

## 🚨 Emergency Helplines (India)

| Service | Number |
|---|---|
| Emergency | **112** |
| Women in distress | **181** |
| Child helpline | **1098** |
| Online financial fraud | **1930** |
| Free legal aid (NALSA) | **15100** |

---

## 🤝 Contributing

Issues and pull requests are welcome. Ideas: more offence categories, an IPC → BNS mapping, additional languages, and an evaluation set for tuning the retrieval distance threshold.

## 📄 License

No license has been specified yet. Add a `LICENSE` file to define how others may use this project.