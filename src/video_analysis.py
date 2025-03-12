import cv2
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

def extract_frames_from_video(video_path, step, by_time=True):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_step = step * fps if by_time else step
    
    frames = []
    frame_idx = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break 

        if frame_idx % frame_step == 0:
            frames.append(frame)
        
        frame_idx += 1

    vid.release()
    return frames


def load_vlm_model(model_type="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_type)

    # Load model with int4 quantization
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_type,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.to(device)

    return processor, model


def load_text_embedder_model(model_type="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_type)


def get_description(
        image,
        processor,
        model,
        prompt="[INST] <image>\n Describe the surroundings of the person but not the person themselves in two to three sentences. Where they are?[/INST]",
        ):
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0][2:], skip_special_tokens=True)


def get_text_embedding(text, embedder):
    return embedder.encode(text, normalize_embeddings=True)


def compare_embeddings(emb1, emb2):
    return util.cos_sim(emb1, emb2)
