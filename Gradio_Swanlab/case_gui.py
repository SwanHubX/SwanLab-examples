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




css = """
h1 {
    text-align: center;
    display:block;
}
"""
title = "Swanlab_Gui"
with gr.Blocks(css=css) as gui:
    gr.Markdown("# title")
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
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