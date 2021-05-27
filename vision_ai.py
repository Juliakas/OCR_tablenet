from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
import io
import cv2
import json


def detect_text(img_np):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=cv2.imencode('.jpg', img_np)[1].tostring())
    response = client.text_detection(image=image)
    with open('./response.json', 'w') as f:
        f.write(AnnotateImageResponse.to_json(response))

    if (len(response.full_text_annotation.pages) > 1):
        raise Exception('Pages > 1')
    elif (len(response.full_text_annotation.pages) == 0):
        return []
    blocks = response.full_text_annotation.pages[0].blocks
    text_in_cells = []

    for block in blocks:
        if len(block.paragraphs) > 1:
            raise Exception('Paragraphs > 1')
        words = []
        for word in block.paragraphs[0].words:
            words.append(''.join(symbol.text for symbol in word.symbols))
        text_in_cells.append(' '.join(words))

    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors'.format(
                            response.error.message))
    # print(text_in_cells)
    # cv2.waitKey()
    return text_in_cells


# detect_text('data/Marmot_data/10.1.1.1.2013_64.bmp')