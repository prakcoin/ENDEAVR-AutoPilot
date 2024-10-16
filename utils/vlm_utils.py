from PIL import Image
import numpy as np
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

def generate_prompt(hlc, speed, light, steer, brake, throttle):
    prompt = f"""You are a powerful vehicle control assistant. Analyze the following sensor data from two RGB cameras in the CARLA Simulator (default and wide lenses) along with additional context data. The control signals given below were generated by an end-to-end convolutional neural network.\n\nCurrent high-level command (HLC): {hlc} ('0' = follow lane, '1' = left turn at intersection, '2' = right turn at intersection, '3' = straight at intersection). Current speed: {speed} km/h. Current traffic light status: {light} (0 = vehicle not at light, 1 = red, 2 = green, 3 = yellow). The predicted control signals generated from the end-to-end convolutional neural network are as follows:\n- Steering angle: {steer} (range: -1.0 to 1.0)\n- Brake value: {brake} (range: 0.0 to 1.0)\n- Throttle value: {throttle} (range: 0.0 to 1.0)\n\nYour task is to evaluate whether these control signals are appropriate for the given driving situation based on the sensor data. If the control signals seem inaccurate or unsafe, adjust them accordingly and explain your corrections. Otherwise, confirm the control signals as correct and explain why they are appropriate. Only use information that can be confidently determined from the provided data and context.\nPlease provide your response in the following format:\n\nControl values:\nSteering Angle: [your_value], Brake: [your_value], Throttle: [your_value]\n\nExplanation:\n[Provide a clear and detailed explanation of why the control signals are appropriate or why adjustments were necessary, based on the sensor data and context. Avoid using list format.]"""
    return prompt

def vlm_inference(main_img, wide_img, hlc, speed, light, steer, brake, throttle):
    model = 'prakcoin/ENDEAVR-AutoPilot-VL-8B'
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

    prompt = generate_prompt(hlc, speed, light, steer, brake, throttle)

    main_img = Image.fromarray(main_img.astype('uint8'), 'RGB')
    wide_img = Image.fromarray(wide_img.astype('uint8'), 'RGB')

    images = [main_img, wide_img]
    response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n{prompt}', images))
    print(response.text)