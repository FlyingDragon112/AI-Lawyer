"""
app.py - Streamlit front end for chatbot.py.

Run:  streamlit run app.py        (in a notebook:  !streamlit run app.py)
"""
import uuid

import streamlit as st

from src.chatbot import chat, get_state

st.set_page_config(page_title="AI Legal Helper (India)", page_icon="⚖️", layout="wide")

GREETING = (
    "Hello! I can help you with problems like **threats, cheating or fraud, theft, or assault**. "
    "Tell me what happened in your own words (English, Hindi or Hinglish are fine) - what happened, "
    "when, and what proof you have. I'll help you understand where you stand and prepare what to do next.\n\n"
    "नमस्ते! अपनी समस्या अपनी भाषा में बताइए।\n\n"
    "*I'm an AI, not a lawyer. For serious or urgent matters, please speak to a lawyer.*"
)
EXAMPLES = [
    "Someone keeps threatening to kill me on WhatsApp",
    "A person took ₹50,000 from me on a false promise and vanished",
    "My landlord is not returning my security deposit",
]
STATUS_TEXT = {
    "match": ("success", "Checked against the Bharatiya Nyaya Sanhita - relevant section(s) found."),
    "weak": ("warning", "Only a partial match in the BNS - treat this as uncertain and ask a lawyer."),
    "none": ("warning", "No BNS section fit these facts, so no section was cited."),
    "out_of_scope": ("info", "Not a criminal matter - I have no verified legal source for it, so no law was cited."),
}
FIT_ICON = {"strong": "🟢 strong fit", "possible": "🟡 possible fit"}
EVIDENCE_ICON = {"have": "✅ have", "can_get": "🟡 can get", "dont_have": "❌ don't have", "unknown": "❔ unknown"}


# ---- session ---------------------------------------------------------------
def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pop("pending", None)


if "thread_id" not in st.session_state:
    new_conversation()


def queue(text):
    st.session_state.pending = text


# ---- header ----------------------------------------------------------------
st.title("⚖️ AI Legal Helper")
st.caption("Everyday criminal-law problems in India (Bharatiya Nyaya Sanhita) · Not legal advice · Not a lawyer")

state = get_state(st.session_state.thread_id)
if state["mode"] == "practice":
    st.info("🎭 **Practice mode** - I'm playing the other side so you can rehearse. "
            "Type **stop** when you want feedback.")

# ---- conversation ----------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(GREETING)
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", on_click=queue, args=(ex,))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Describe your problem...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply = chat(prompt, st.session_state.thread_id)
        except Exception as e:  # missing API key, network, quota...
            reply = f"⚠️ Sorry, something went wrong: {e}"
        st.markdown(reply)
    # saved in the history so it survives the rerun below (which refreshes the sidebar)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ---- sidebar (rendered last so it always shows the freshest state) ----------
with st.sidebar:
    st.header("Your case so far")
    case = state["case"]
    if not case or not case.get("summary"):
        st.caption("Nothing yet - it fills in as you tell me what happened.")
    else:
        st.markdown(f"**Summary:** {case['summary']}")
        if case.get("other_party"):
            st.markdown(f"**Other party:** {case['other_party']}")
        if case.get("place"):
            st.markdown(f"**Place:** {case['place']}")
        if case.get("harm"):
            st.markdown(f"**Harm / loss:** {case['harm']}")
        if case.get("timeline"):
            st.markdown("**Timeline**")
            st.table([{"When": e["when"], "What happened": e["what"]} for e in case["timeline"]])
        if case.get("evidence"):
            st.markdown("**Proof**")
            for e in case["evidence"]:
                st.markdown(f"- {e['item']} - {EVIDENCE_ICON.get(e['status'], e['status'])}")
        if case.get("steps_already_taken"):
            st.markdown("**Already done**")
            for s in case["steps_already_taken"]:
                st.markdown(f"- {s}")
        if case.get("missing_info"):
            st.markdown("**Still needed**")
            for s in case["missing_info"]:
                st.markdown(f"- {s}")

    if state["status"] in STATUS_TEXT:
        st.divider()
        st.subheader("What the answer was checked against")
        kind, text = STATUS_TEXT[state["status"]]
        getattr(st, kind)(text)
        for s in state["sections"]:
            st.markdown(f"**BNS Section {s['section']}** - {s['name']}  \n"
                        f"{FIT_ICON.get(s['fit'], s['fit'])}: {s['reason']}")
        for l in state["lookups"]:
            if l["exists"] is False:
                st.warning(f"Section {l['number']} does not exist in the BNS.")

    st.divider()
    st.subheader("Do more")
    st.button("📝 Draft my complaint / letter", use_container_width=True, on_click=queue,
              args=("Please draft the complaint or letter I need, using what I've told you.",))
    st.button("🎭 Practice with the other side", use_container_width=True, on_click=queue,
              args=("I want to practise the conversation with the other side.",))
    st.button("🛑 Stop practice & give feedback", use_container_width=True, on_click=queue,
              args=("Stop the practice and give me feedback.",), disabled=state["mode"] != "practice")
    st.button("➕ New conversation", use_container_width=True, on_click=new_conversation)

    with st.expander("🚨 In danger or need urgent help?"):
        st.markdown("Call **112** (emergency) · Women in distress **181** · Child helpline **1098** · "
                    "Online financial fraud **1930** · Free legal aid **15100**")