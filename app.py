# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import gradio as gr

# pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

model_path = "Model/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# text="Why You Can Trust Forbes Advisor Small Business \
# The Forbes Advisor Small Business team is committed to bringing you unbiased \
# rankings and information with full editorial independence. We use product data, \
# strategic methodologies and expert insights to inform all of our content and guide \
# you in making the best decisions for your business journey.\
# We reviewed 11 systems to help you find the best blogging platform for your blog or \
# small business. Our ratings looked at factors that included the platform’s starting \
# price (including whether it offered a free trial or free version); useful general features,\
# such as drag-and-drop functionality and search engine optimization (SEO) tools; unique features, \
# how well the blogging platform fared on third-party review sites and a final review by our experts.\
# All ratings are determined solely by our editorial team."

# print(text_summary(text)[0])

def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary,inputs="text",outputs='text',title='Text Summarization Gradio Huggingface')
demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarization", lines=6)],
                    outputs=[gr.Textbox(label="Summarized text", lines=4)],
                    title='Text Summarization',
                    description='This application will be used to summarize the text',
                    concurrency_limit=16)

demo.launch(share=True)