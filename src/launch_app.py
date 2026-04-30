from legal_rag import LegalRAGBot

import gradio as gr
from typing import Dict, List

def launch_app():
    bot = LegalRAGBot()

    def build_source_lines(citations: List[Dict[str, str]]) -> List[str]:
        source_lines = []
        seen_sources = set()
        for item in citations:
            brief = item["source_brief"].strip()
            if not brief or brief in seen_sources:
                continue
            seen_sources.add(brief)
            source_lines.append(f"- {brief}")
            if len(source_lines) >= 3:
                break
        return source_lines

    def chat_interface(message, history):
        history = history or []
        message = (message or "").strip()
        if not message:
            yield history, ""
            return

        paired_history = []
        if history:
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    paired_history.append((history[i]["content"], history[i + 1]["content"]))

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history, ""

        final_answer = ""
        final_citations: List[Dict[str, str]] = []
        for partial_answer, citations in bot.stream_answer(message, paired_history):
            final_answer = partial_answer
            final_citations = citations
            history[-1]["content"] = partial_answer
            yield history, ""

        source_lines = build_source_lines(final_citations)
        citation_text = "\n\nTham khảo thêm:\n" + "\n".join(source_lines) if source_lines else ""
        # history[-1]["content"] = final_answer + citation_text
        history[-1]["content"] = final_answer
        yield history, ""

    with gr.Blocks(title="Chatbot Luật Việt Nam", css="footer {visibility: hidden}") as demo:
        gr.Markdown("### Chatbot văn bản luật Việt Nam")
        gr.Markdown("*RAG: ChromaDB + AITeamVN Embedding/Reranker + Groq LLM*")

        chatbot = gr.Chatbot(height=500, label="Trợ lý pháp lý")
        with gr.Row():
            msg = gr.Textbox(
                label="Câu hỏi pháp lý",
                placeholder="Ví dụ: Mức phạt xe máy không đăng ký năm 2024 là bao nhiêu?",
                scale=4,
                submit_btn=True,
            )
            gr.ClearButton([msg, chatbot], value="Xóa")

        gr.Examples(
            examples=[
                "Mức phạt xe máy không đăng ký năm 2024 là bao nhiêu?",
                "Phạt nồng độ cồn xe máy năm 2024 theo quy định nào?",
                "Không mang giấy phép lái xe bị phạt thế nào?",
            ],
            inputs=msg,
        )

        msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])

    demo.launch()