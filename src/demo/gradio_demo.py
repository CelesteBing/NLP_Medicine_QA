import gradio as gr
from src.ner.run_ner import ner_extract
from src.qa.run_qa import rag_answer


def process_ner(text):
    return ner_extract(text)


def process_qa(question):
    answer, evidence = rag_answer(question)
    return answer, evidence


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Medical NLP Demo")
    gr.Markdown("Test the NER and QA modules of the biomedical NLP system.")

    with gr.Tab("NER Demo"):
        gr.Markdown("Enter a medical text to test named entity recognition.")
        with gr.Row():
            with gr.Column():
                ner_input = gr.Textbox(
                    label="Input medical text",
                    placeholder="For example: The patient has headache and takes aspirin.",
                    lines=5
                )
                ner_button = gr.Button("Extract Entities", variant="primary")

            with gr.Column():
                ner_output = gr.HighlightedText(
                    label="NER Result",
                    color_map={"Drug": "green", "Disease": "red"}
                )

        ner_button.click(
            fn=process_ner,
            inputs=ner_input,
            outputs=ner_output
        )

    with gr.Tab("QA Demo"):
        gr.Markdown("Enter a biomedical question to test retrieval-augmented QA.")
        with gr.Row():
            with gr.Column():
                qa_input = gr.Textbox(
                    label="Question",
                    placeholder="For example: Does mitochondria play a role in remodelling lace plant leaves?",
                    lines=2
                )
                qa_button = gr.Button("Get Answer", variant="primary")

            with gr.Column():
                qa_answer = gr.Textbox(label="Answer", lines=4)
                qa_evidence = gr.Textbox(label="Evidence", lines=8)

        qa_button.click(
            fn=process_qa,
            inputs=qa_input,
            outputs=[qa_answer, qa_evidence]
        )


if __name__ == "__main__":
    demo.launch()