import os
import aiofiles
from typing import List
from fastapi import FastAPI, File, UploadFile
from models.neural_style_transfer import NeuralStyleTransfer
from utils.config import load_config

app = FastAPI()
config = load_config()
nst = NeuralStyleTransfer(config)

@app.post("/style_transfer/")
async def style_transfer(content_image: UploadFile = File(...), style_images: List[UploadFile] = File(...)):
    # Create the 'data/tmp' directory if it doesn't exist
    tmp_dir = "data/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    content_image_path = f"{tmp_dir}/{content_image.filename}"
    style_images_paths = [f"{tmp_dir}/{style_image.filename}" for style_image in style_images]

    async with aiofiles.open(content_image_path, "wb") as content_file:
        content = await content_image.read()
        await content_file.write(content)

    for idx, style_image in enumerate(style_images):
        style_image_path = style_images_paths[idx]
        async with aiofiles.open(style_image_path, "wb") as style_file:
            style = await style_image.read()
            await style_file.write(style)

    config["content_img_name"] = content_image.filename
    config["style_img_name"] = [style_image.filename for style_image in style_images]
    dump_path, _ = nst.run(content_image_path, style_images_paths)

    output_image_path = os.path.join(dump_path, nst.generate_out_img_name())
    return {"output_image_path": output_image_path}