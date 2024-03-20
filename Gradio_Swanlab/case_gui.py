import gradio as gr
import argparse
import sys
import os

from swanlab_gui import(
    gradio_swanlab,
    start_swanlab,
    stop_swanlab,
)
import swanlab





title = "<h1 style='font-size: 40px;'><center>Swanlab_Gui</center></h1>"
with gr.Blocks() as gui:
    gr.Markdown(title)
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
    pass
    
    
    
    
    
    
if __name__ == "__main__":
    gui.launch()