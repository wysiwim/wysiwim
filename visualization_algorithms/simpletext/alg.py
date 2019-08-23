import os
import sys
from PIL import Image, ImageDraw, ImageFont

def load_code(path):
    code = []
    with open(path, "r") as fin:
        for l in fin:
            code.append(l)
    return "".join(code)

def text2png(code, lang):
    code = code.replace("\t", "    ")
    background = (255, 255, 255)

    fontsize = 14
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', fontsize)

    width, height = ImageDraw.Draw(Image.new('RGBA', (1,1), background)).textsize(code, font)
    image = Image.new('RGBA', (int(width * 1.1), int(height * 1.1)), background)
    draw = ImageDraw.Draw(image)
    draw.text((10, 2), code, fill='black', font=font)
    return image

def from_to_file(in_path, out_path):
    code = load_code(in_path)
    image = text2png(code)
    image.save(out_path, dpi=(600, 600))

if __name__ == "__main__":
    from_to_file("<path>/example.java", "<path>/example_simple_text.png")
