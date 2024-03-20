import gradio as gr
import argparse
import sys
import os
from  folder_class import Folders


from swanlab_gui import(
    gradio_swanlab,
    start_swanlab,
    stop_swanlab,
)
import swanlab


headless=False



css = """
h1 {
    text-align: center;
    display:block;
}
"""

if os.path.exists("./style.css"):
    with open(os.path.join("./style.css"), "r", encoding="utf8") as file:
        css += file.read() + "\n"
title = "Swanlab_Gui"
with gr.Blocks(css=css) as gui:
    gr.Markdown(title,elem_id='title')
    
    
    with gr.Tab("Folders"):
        folders = Folders(headless=headless)
        
        with gr.Row():
            button_run = gr.Button("Start training", variant="primary")

            button_stop_training = gr.Button("Stop training")
        with gr.Row():
         # Setup gradio swanlab buttons
            (
                button_start_swanlab,
                button_stop_swanlab,
            ) = gradio_swanlab()

            button_start_swanlab.click(
                start_swanlab,
            # inputs=[dummy_headless, folders.logging_dir],
                show_progress=False,
            )

            button_stop_swanlab.click(
                stop_swanlab,
                show_progress=False,
            )
        
    
    with gr.Tab("Parameters"):
        
    
        with gr.Row():
            button_run = gr.Button("Start training", variant="primary")

            button_stop_training = gr.Button("Stop training")
        
        
  
    pass
    
    
    
    
    
    
if __name__ == "__main__":
    gui.launch()