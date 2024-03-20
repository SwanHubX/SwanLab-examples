import gradio as gr
import os
import sys
from tkinter import filedialog, Tk


folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

ENV_EXCLUSION = ["COLAB_GPU", "RUNPOD_POD_ID"]

def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', "")

    return file_path

def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)


def get_folder_path(folder_path=""):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_folder_path = folder_path

        initial_dir, initial_file = get_dir_and_file(folder_path)

        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        folder_path = filedialog.askdirectory(initialdir=initial_dir)
        root.destroy()

        if folder_path == "":
            folder_path = current_folder_path

    return folder_path

class Folders:
    def __init__(self, headless=False):
        self.headless = headless

        with gr.Row():
            self.train_data_dir = gr.Textbox(
                label='Image folder',
                placeholder='Folder where the training folders containing the images are located',
            )
            self.train_data_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not self.headless)
            )
            self.train_data_dir_folder.click(
                get_folder_path,
                outputs=self.train_data_dir,
                show_progress=False,
            )
            self.reg_data_dir = gr.Textbox(
                label='Regularisation folder',
                placeholder='(Optional) Folder where where the regularization folders containing the images are located',
            )
            self.reg_data_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not self.headless)
            )
            self.reg_data_dir_folder.click(
                get_folder_path,
                outputs=self.reg_data_dir,
                show_progress=False,
            )
        with gr.Row():
            self.output_dir = gr.Textbox(
                label='Output folder',
                placeholder='Folder to output trained model',
            )
            self.output_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not self.headless)
            )
            self.output_dir_folder.click(
                get_folder_path,
                outputs=self.output_dir,
                show_progress=False,
            )
            self.logging_dir = gr.Textbox(
                label='Logging folder',
                placeholder='Optional: enable logging and output TensorBoard log to this folder',
            )
            self.logging_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not self.headless)
            )
            self.logging_dir_folder.click(
                get_folder_path,
                outputs=self.logging_dir,
                show_progress=False,
            )
        with gr.Row():
            self.output_name = gr.Textbox(
                label='Model output name',
                placeholder='(Name of the model to output)',
                value='last',
                interactive=True,
            )
            self.training_comment = gr.Textbox(
                label='Training comment',
                placeholder='(Optional) Add training comment to be included in metadata',
                interactive=True,
            )
        self.train_data_dir.blur(
            remove_doublequote,
            inputs=[self.train_data_dir],
            outputs=[self.train_data_dir],
        )
        self.reg_data_dir.blur(
            remove_doublequote,
            inputs=[self.reg_data_dir],
            outputs=[self.reg_data_dir],
        )
        self.output_dir.blur(
            remove_doublequote,
            inputs=[self.output_dir],
            outputs=[self.output_dir],
        )
        self.logging_dir.blur(
            remove_doublequote,
            inputs=[self.logging_dir],
            outputs=[self.logging_dir],
        )
