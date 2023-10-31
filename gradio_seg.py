import gradio as gr
from inference import load_seg_model, get_palette, generate_mask
import cv2
import numpy as np
import os

device = 'cpu'


def algorithm_improve_seg(image, mode_expansion, mode_smoothing, factor_kernel: int = 5, smoothing: bool = True,
                          factor_smooth: int = 5):
    # image = cv2.imread(image)
    assert image is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((factor_kernel, factor_kernel), np.uint8)

    remove_noise_bg_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    output = remove_noise_bg_img

    if mode_expansion == "erosion":
        output = cv2.erode(remove_noise_bg_img, kernel, iterations=1)
    elif mode_expansion == "dilate":
        output = cv2.dilate(remove_noise_bg_img, kernel, iterations=1)
    else:
        pass
    if smoothing is True:
        if mode_smoothing == "blur":
            output = cv2.blur(output, (factor_smooth, factor_smooth))
        elif mode_smoothing == "gaussian blur":
            output = cv2.GaussianBlur(output, (factor_smooth, factor_smooth), 0)
        elif mode_smoothing == "median blur":
            output = cv2.medianBlur(output, 5)
        else:
            pass
    return output


def initialize_and_load_models():
    checkpoint_path = 'model/unet_cloth_seg.pth'
    net = load_seg_model(checkpoint_path, device=device)
    return net


net = initialize_and_load_models()
palette = get_palette(4)


def run(image, mode_expansion, mode_smoothing, factor_kernel, smoothing, factor_smooth):
    cloth_seg = generate_mask(image, net=net, palette=palette, device=device)
    output = algorithm_improve_seg(cloth_seg, mode_expansion=mode_expansion, mode_smoothing=mode_smoothing,
                                   factor_kernel=factor_kernel,
                                   smoothing=smoothing, factor_smooth=factor_smooth)
    return output


# -----------------DEMO USE INTERFACE GRADIO --------------------------------------------------#
# input_image = gr.Image(source='upload', type="numpy")
input_image = gr.inputs.Image(label="Input Image", type="pil")
with gr.Accordion("Advanced options", open=False):
    mode_expansion = gr.Dropdown(["erode", "dilate"], label="Expansion",
                                 info="dilation or erosion of clothes segmentation.", value="dilate")
    factor_kernel = gr.Slider(label="Control factor strength", minimum=1, maximum=6, value=5, step=1)
    mode_smoothing = gr.Dropdown(["blur", "gaussian blur", "median blur"], label="Smoothing",
                                 info=" smoothing image and remove anti-aliaing.", value="gaussian blur")
    smoothing = gr.Checkbox(label="Smoothing mode", value=True)
    factor_smooth = gr.Slider(label="Control factor smooth", minimum=1, maximum=7, value=5, step=1)

result_gallery = gr.outputs.Image(label="Cloth Segmentation", type="pil")

ips = [input_image, mode_expansion, mode_smoothing, factor_kernel, smoothing, factor_smooth]
outputs = [result_gallery]

title = "Demo for Garment Segmentation"
description = "<p align='center'>This is demo for clothes segmentation and used algorithm remove noise and anti-aliasing to improve quality.</p>"

gr.Interface(fn=run, inputs=ips, outputs=outputs, title=title, description=description, examples=[
    [os.path.join(os.path.dirname(__file__), "input/img1.jpg")],
    [os.path.join(os.path.dirname(__file__), "input/img2.jpg")],
    [os.path.join(os.path.dirname(__file__), "input/img3.jpg")],
    [os.path.join(os.path.dirname(__file__), "input/img4.jpeg")],
]).launch(share=True)
