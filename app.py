"""
Step 6b: a small Gradio web interface for the Company Assistant.

  python app.py      then open http://127.0.0.1:7860

How it works (the important part):
  - Every message from the user starts a NEW conversation thread (a fresh thread_id).
  - We call review_graph.invoke(...) with review=True. If the graph pauses in the "review"
    node, we show the review panel. The thread_id is kept in a gr.State so the "Save" /
    "Reject" buttons can RESUME that exact paused run (Command(resume=...)).
  - If the graph does not pause (chitchat, saved questions, failed fetch), we just show the answer.

This uses REAL websites, the REAL LLM and your REAL database (chroma_db).
"""
import uuid

import gradio as gr
from langgraph.types import Command

from step3_graph import review_graph


# ---------- talking to the graph ----------
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _run(graph_input, thread_id: str):
    """Run (or resume) the graph. Returns (answer, None) when it finished,
    or (None, payload) when it paused and is waiting for a human."""
    config = _config(thread_id)
    result = review_graph.invoke(graph_input, config)
    snapshot = review_graph.get_state(config)
    if snapshot.next:                                    # paused: something is waiting to run
        return None, snapshot.tasks[0].interrupts[0].value
    return result["answer"], None


# ---------- what to show on the screen ----------
def _info_markdown(payload: dict) -> str:
    warnings = payload.get("warnings") or []
    warning_lines = "\n".join(f"- WARNING: {w}" for w in warnings) if warnings else "- no automatic warnings"
    return (
        f"**URL the page really ended up on:** {payload['url']}  \n"
        f"**Page title:** {payload.get('page_title')}\n\n"
        f"**Automatic checks:**\n{warning_lines}\n\n"
        f"**Start of the page text (compare it with the summary):**\n```\n{payload['source_excerpt']}\n```"
    )


def _show(answer, payload):
    """Return the 5 values for: answer box, review panel, name box, summary box, info text."""
    if payload:
        return (gr.Textbox(value="Paused: please review the proposed company below."),
                gr.Column(visible=True),
                gr.Textbox(value=payload["name"]),
                gr.Textbox(value=payload["summary"]),
                gr.Markdown(value=_info_markdown(payload)))
    return (gr.Textbox(value=answer),
            gr.Column(visible=False),
            gr.Textbox(value=""),
            gr.Textbox(value=""),
            gr.Markdown(value=""))


# ---------- button handlers ----------
def on_send(user_text, old_thread_id):
    user_text = (user_text or "").strip()
    if not user_text:
        return (old_thread_id,) + _show("Type a company URL or a question first.", None)
    thread_id = str(uuid.uuid4())                        # new message = new conversation thread
    try:
        answer, payload = _run({"user_text": user_text, "review": True}, thread_id)
    except Exception as e:                               # show the problem instead of a blank screen
        return ("",) + _show(f"Error: {type(e).__name__}: {e}", None)
    return (thread_id if payload else "",) + _show(answer, payload)


def _resume(thread_id, decision):
    if not thread_id:
        return ("",) + _show("Nothing is waiting for review.", None)
    try:
        answer, payload = _run(Command(resume=decision), thread_id)
    except Exception as e:
        return ("",) + _show(f"Error: {type(e).__name__}: {e}", None)
    return ("",) + _show(answer, payload)


def on_save(thread_id, name, summary):
    # Whatever is in the two boxes is saved. If you did not change them, this is an approval.
    return _resume(thread_id, {"action": "edit", "name": name, "summary": summary})


def on_reject(thread_id, name, summary):
    return _resume(thread_id, {"action": "reject"})


# ---------- layout ----------
with gr.Blocks(title="Company Assistant") as demo:
    gr.Markdown(
        "# Company Assistant\n"
        "Send a company URL to fetch and save it (you review it first), ask a question about "
        "your saved companies, or just chat."
    )
    thread_state = gr.State("")        # thread_id of the run that is waiting for review (if any)

    with gr.Row():
        user_box = gr.Textbox(label="Your message", placeholder="Tell me about https://example.com", scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)

    with gr.Column(visible=False) as review_panel:
        gr.Markdown("### Review before saving")
        info_md = gr.Markdown()
        name_box = gr.Textbox(label="Company name (you can edit)")
        summary_box = gr.Textbox(label="Summary (you can edit)", lines=4)
        with gr.Row():
            save_btn = gr.Button("Save", variant="primary")
            reject_btn = gr.Button("Reject", variant="stop")

    outputs = [thread_state, answer_box, review_panel, name_box, summary_box, info_md]
    send_btn.click(on_send, inputs=[user_box, thread_state], outputs=outputs)
    user_box.submit(on_send, inputs=[user_box, thread_state], outputs=outputs)
    save_btn.click(on_save, inputs=[thread_state, name_box, summary_box], outputs=outputs)
    reject_btn.click(on_reject, inputs=[thread_state, name_box, summary_box], outputs=outputs)


if __name__ == "__main__":
    demo.launch()