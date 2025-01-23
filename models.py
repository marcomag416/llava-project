from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

class qwen2vl():
    def __init__(self, flash_attention=False, min_resolution=256, max_resolution=512):
        if flash_attention:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto", offload_buffers=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto", offload_buffers=True)

        min_pixels = min_resolution*28*28
        max_pixels = max_resolution*28*28 
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def infer(self, imgs, prompts):
        chats = [[
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                    },
                    {
                        "type":"text",
                        "text":prompt
                    }
                ]
            }
        ] for prompt in prompts]

        # Preprocess the inputs
        text_prompts = [self.processor.apply_chat_template(chat, add_generation_prompt=True) for chat in chats]
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

        inputs = self.processor(text=text_prompts, images=imgs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=8)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return output_text

        