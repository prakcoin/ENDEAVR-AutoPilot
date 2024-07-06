import carla
import torch
import numpy as np
from .sensors import RGBCamera
from model.AVModel import AVModel
from torchvision.transforms import v2

class CropCustom(object):
    def __call__(self, img):
        width, height = img.size
        top = int(height / 2.05)
        bottom = int(height / 1.05)
        cropped_img = img.crop((0, top, width, bottom))
        return cropped_img
    
preprocess = v2.Compose([
    v2.ToPILImage(),
    CropCustom(),
    v2.Resize((59, 128)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=(0.4872, 0.4669, 0.4469,), std=(0.1138, 0.1115, 0.1074,)),
])

def load_model(model_path, device):
    model = AVModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def to_rgb(image):
    image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    image_array = image_array[:, :, :3]
    image_array = image_array[:, :, ::-1]
    image_array = image_array.copy()
    return image_array

def model_control(sensor, model):
    image = sensor.get_sensor_data()
    image_array = to_rgb(image)
    input_tensor = preprocess(image_array).unsqueeze(0)

    #after_tensor = preprocess(array)
    #plt.imshow(after_tensor.permute(1, 2, 0))
    #plt.title('After Norm Image Tensor')
    #plt.axis('off')
    #plt.show()

    with torch.no_grad():
        output = model(input_tensor)
    
    output = output.detach().cpu().numpy().flatten()
    steer, throttle_brake = output

    if throttle_brake >= 0.5:
        throttle = (throttle_brake - 0.5) / 0.5
    else:
        brake = (0.5 - throttle_brake) / 0.5
    
    steer = (float(steer) * 2.0) - 1.0
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

def start_camera(world, vehicle):
    rgb_cam = RGBCamera(world, vehicle)
    return rgb_cam