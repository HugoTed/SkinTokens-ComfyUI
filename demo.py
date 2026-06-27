import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"
gr.TEMP_DIR = "tmp_gradio"

from src.pipeline import (
    collect_files,
    export_suffix,
    load_model,
    map_output_path,
    run_rig,
    start_bpy_server,
    wait_for_bpy_server,
)

MODEL_CKPTS = [
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
]

HF_PATHS = [
    "None",
]

TOT = 0


def run_gradio(
    files,
    top_k,
    top_p,
    temperature,
    repetition_penalty,
    num_beams,
    use_skeleton,
    use_transfer,
    use_postprocess,
    export_format,
    model_ckpt,
    hf_path,
):
    if not files:
        return "Please upload at least one 3D model.", None

    tmp_out = Path(tempfile.mkdtemp(prefix="tokenrig_"))
    filepaths = [Path(f.name) for f in files]
    suffix = export_suffix(export_format)
    global TOT
    outputs = []
    for filepath in filepaths:
        TOT += 1
        outputs.append(tmp_out / f"res_{TOT}{suffix}")

    run_rig(
        filepaths,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        num_beams,
        use_skeleton,
        use_transfer,
        use_postprocess,
        outputs,
        model_ckpt,
        hf_path,
    )

    return f"Processed {len(outputs)} models.", [str(p) for p in outputs]


def launch_gradio():
    model_ckpts = MODEL_CKPTS
    hf_paths = HF_PATHS
    default_ckpt = model_ckpts[0] if model_ckpts else ""
    default_hf = hf_paths[0] if hf_paths else "None"

    with gr.Blocks(title="TokenRig Demo") as demo:
        gr.Markdown("## TokenRig Demo")
        gr.Markdown("Upload 3D assets, configure parameters, generate rigged GLB or FBX")

        files = gr.File(
            label="3D Models",
            file_count="multiple",
            file_types=[".obj", ".fbx", ".glb"],
        )

        with gr.Accordion("Generation Parameters", open=True):
            model_ckpt = gr.Dropdown(
                choices=model_ckpts,
                value=default_ckpt,
                label="Model checkpoint",
                interactive=True,
            )
            hf_path = gr.Dropdown(
                choices=hf_paths,
                value=default_hf,
                label="HF path",
                interactive=True,
            )
            top_k = gr.Slider(1, 200, value=5, step=1, label="top_k")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="temperature")
            repetition_penalty = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="repetition_penalty")
            num_beams = gr.Slider(1, 20, value=10, step=1, label="num_beams")
            use_skeleton = gr.Checkbox(False, label="Use skeleton (only generate skin if skeleton exists)")
            use_transfer = gr.Checkbox(False, label="Use transfer (maintain texture)")
            use_postprocess = gr.Checkbox(False, label="Use postprocess (voxel skin)")
            export_format = gr.Dropdown(
                choices=["glb", "fbx"],
                value="glb",
                label="Export format",
                interactive=True,
            )

        run_btn = gr.Button("Run", variant="primary")
        load_btn = gr.Button("Load Model")
        log = gr.Textbox(label="Status")
        output = gr.File(label="Generated mesh", interactive=False)

        load_btn.click(
            lambda ckpt, hf: load_model(ckpt, hf)[0],
            inputs=[model_ckpt, hf_path],
            outputs=[log],
        )

        run_btn.click(
            run_gradio,
            inputs=[
                files,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                num_beams,
                use_skeleton,
                use_transfer,
                use_postprocess,
                export_format,
                model_ckpt,
                hf_path,
            ],
            outputs=[log, output],
        )

    demo.launch(server_port=1024)


def run_cli(args):
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    files = collect_files(input_path)
    if not files:
        raise RuntimeError("No valid 3D files found.")

    if len(files) == 1 and output_path.suffix:
        outputs = [output_path]
    else:
        outputs = [
            map_output_path(f, input_path, output_path, export_format=args.export_format)
            for f in files
        ]

    run_rig(
        files,
        args.top_k,
        args.top_p,
        args.temperature,
        args.repetition_penalty,
        args.num_beams,
        args.use_skeleton,
        args.use_transfer,
        args.use_postprocess,
        outputs,
        args.model_ckpt,
        args.hf_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenRig Demo")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=10)

    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_transfer", action="store_true")
    parser.add_argument("--use_postprocess", action="store_true")
    parser.add_argument("--export_format", choices=["glb", "fbx"], default="glb")

    parser.add_argument("--model_ckpt", default=MODEL_CKPTS[0] if MODEL_CKPTS else "")
    parser.add_argument("--hf_path", default=None)

    parser.add_argument("--gradio", action="store_true")

    args = parser.parse_args()

    start_bpy_server()
    wait_for_bpy_server()

    if args.gradio or not args.input:
        launch_gradio()
    else:
        run_cli(args)
