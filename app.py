# Original product reference: https://huggingface.co/spaces/fffiloni/Image-to-Story

import gradio as gr
import re
import os
import time

hf_token = os.environ.get('HF_TOKEN')
# Monitor imports as suggested by Claude Opus 5.5 with Medium thinking in conjunction with the monitor.py script.
import monitor
monitor.start_monitor()

from gradio_client import Client, handle_file

clipi_client = Client("fffiloni/CLIP-Interrogator-2")

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import InferenceClient

from prometheus_client import start_http_server, Counter, Summary

# --- Default Prometheus metrics - should change to better suit our product ---
REQUEST_COUNTER = Counter('app_requests_total', 'Total number of requests')
SUCCESSFUL_REQUESTS = Counter('app_successful_requests_total', 'Total number of successful requests')
FAILED_REQUESTS = Counter('app_failed_requests_total', 'Total number of failed requests')
REQUEST_DURATION = Summary('app_request_duration_seconds', 'Time spent processing request')

model_path = "Qwen/Qwen2.5-0.5B-Instruct"  # Qwen/Qwen2.5-3B-Instruct was used in Case Study 1, but when prompting Claude Opus 5.5 with "I can't use Qwen2.5-3B-Instruct anymore as that is far too large. What other model should I use that is much smaller in size?", it suggests we switch to the 0.5B version to save space on the VM as the original 3B model is quite large (~6gb)
remote_model_path = "openai/gpt-oss-20b"
# The below system prompt was adopted from our original LLM prompt by Claude Opus 5.5 on Medium thinking mode in order for the models to create safety advice properly.
SYSTEM_PROMPT = (
    "You are a professional safety analyst. The user will give you a description of an image. "
    "Reply with the most important safety advice for the scene in that image.\n\n"
    "Format your reply exactly like this:\n"
    "**<short title>**\n\n"
    "<one paragraph of 2-4 sentences of safety advice>\n\n"
    "Rules: Do not ask questions. Do not use bullet points, lists, or headings. "
    "If the description is vague, give the safety advice that best fits it."
)


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, token=hf_token, torch_dtype="auto", device_map="auto")  # Adjusted with Claude Opus 5.5 to run on CPU as CUDA is no longer available

# Revised by Claude Opus 5.5 on Medium thinking to use Qwen's chat template instead of Ollama's while also providing better support for the local (small) model.
def gen_safety_advice(description, platform):
    """Generate a titled, 2-4 sentence segment of safety advice for an image description."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Image description: {description}"},
    ]
    if platform == "Local (Qwen/Qwen2.5-0.5B-Instruct)":
        return gen_local(messages)
    return gen_remote(messages)


# Revised by Claude Opus 5.5 on Medium thinking to use Qwen's chat template instead of Ollama's.
def gen_local(messages):
    gr.Info('Calling Qwen2.5-0.5B-Instruct (local)...')
    start_time = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    new_tokens = generate_ids[0][inputs["input_ids"].shape[-1]:]  # drop the prompt tokens
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] platform=local elapsed={elapsed:.2f}s")
    return f"{text}\n\n_(Response time - {elapsed:.2f}s)_"


def gen_remote(messages):
    gr.Info('Calling OpenAI/gpt-oss-20b (remote)...')
    start_time = time.perf_counter()
    inf_client = InferenceClient(token=hf_token)
    response = inf_client.chat_completion(model=remote_model_path, messages=messages, max_tokens=4096)  # Revised by Claude Opus 5.5 on Medium thinking so the system message is passed properly, not as raw text inside the user message like before.
    text = response.choices[0].message.content
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] platform=remote elapsed={elapsed:.2f}s")
    return f"{text}\n\n_(Response time - {elapsed:.2f}s)_"


def infer(image_input, text_input, running_platform):
    """Generate 2-4 sentence of the most important safety advice based on an image using CLIP Interrogator and an LLM.
    
    Args:
        image_input: A file path to the input image to analyze.
        running_platform: A string indicating the target running platform for the LLM to run on (local or remote).
    
    Returns:
        A formatted, 2-4 sentence segment of safety advice based on the provided image.

    """
    if running_platform == "":
        gr.Info("No model selected, using remote inference model by default.")

    try:
        gr.Info('Calling CLIP Interrogator ...')

        clipi_result = clipi_client.predict(
            input_image=handle_file(image_input),
            interrogation_mode="best",
            best_mode_max_flavors=4,
            api_name="/clipi2"
        )
        print(clipi_result)
    except Exception as e:
        if text_input != "":
            clipi_result = text_input
        else:
            gr.Info("No text input provided. Please provide a text description and try again.")

    # Simplified by Claude Opus 5.5 on Medium thinking into the below return statement as there was unneeded bloat from the original product on HuggingFace.
    return gen_safety_advice(clipi_result, running_platform)

# The resource-warning css format was suggested by Claude Opus 5.5 with Medium thinking in conjuction with monitor.py.
css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
div#safety_advice {
    font-size: 1.5em;
    line-height: 1.4em;
}
#resource-warning {
    background: #fdecea;
    color: #750707;
    border: 1px solid #f5c2c0;
    border-radius: 8px;
    padding: 12px 16px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center">Image to Safety Advice</h1>
            <p style="text-align: center">Upload an image, get safety advice based on the image content!</p>
            """
        )
        warning_banner = gr.HTML()  # Suggested by Claude Opus 5.5 on Medium thinking in conjunction with monitor.py
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image Input", type="filepath", elem_id="image-in")
                text_input = gr.Textbox(label="(Optional) Image Description as Text", elem_id="text-input")
                running_platform = gr.Radio(label="LLM Model", choices=["Local (Qwen/Qwen2.5-0.5B-Instruct)", "Remote (OpenAI/gpt-oss-20b)"])
                submit_btn = gr.Button('Give me safety advice')
            with gr.Column():
                safety_advice = gr.Markdown(label="Generated Safety Advice", elem_id="safety_advice")  # Textbox was changed to Markdown as suggested by Claude 5.5 on Medium thinking to better format the LLM output.
        
    submit_btn.click(fn=infer, inputs=[image_in, text_input, running_platform], outputs=[safety_advice])
    monitor_timer = gr.Timer(5)  # Suggested by Claude Opus 5.5 on Medium thinking in conjunction with monitor.py
    monitor_timer.tick(fn=monitor.get_warning_html, outputs=[warning_banner], show_progress="hidden")  # Suggested by Claude Opus 5.5 on Medium thinking in conjunction with monitor.py

demo.queue(max_size=12).launch(server_name="0.0.0.0", ssr_mode=False, mcp_server=True)  # Adjusted with Claude Opus 5.5 on Medium thinking when asked how to adjust to run on a vm instead of a HuggingFace space - necessary change to allow Gradio to accept connections outside the VM


# The below code segment is from the class CaseStudy3 GitHub repository.
if __name__ == "__main__":
    # Expose Prometheus metrics at :8000/metrics
    start_http_server(8000)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )