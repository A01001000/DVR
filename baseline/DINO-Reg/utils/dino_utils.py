import torch
import numpy as np

def get_patch_coordinates(idx, image_size, patch_size=(14,14)):
    # Image dimensions
    img_height, img_width = image_size

    # Number of patches along each dimension
    height_patches = img_height // patch_size[0]
    width_patches = img_width // patch_size[1]

    # Row and column in the patch grid
    row = idx // width_patches
    col = idx % width_patches

    # Calculate coordinate range
    x1 = col * patch_size[1]
    y1 = row * patch_size[0]
    x2 = x1 + patch_size[1] - 1
    y2 = y1 + patch_size[0] - 1

    return (x1, y1, x2, y2)


def extract_dinov2_patch_feature(input_array, transform, model, device_id):

    assert len(input_array.shape) == 3  # 2D image

    """flipping the input if needed"""
    # input_array = np.swapaxes(input_array, 0,1)

    input_rgb_array = input_array[np.newaxis, :, :, :]

    input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
    input_tensor = transform(input_tensor)
    feature_array = model.forward_features(input_tensor.to(device=torch.device('cuda', device_id)))[
        'x_norm_patchtokens'].detach().cpu().numpy()
    # "x_norm_clstoken": x_norm[:, 0],
    # "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
    # "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
    # "x_norm": x_norm,
    # "x_prenorm": x,
    # "masks": masks,
    del input_tensor

    return feature_array


def extract_dinov2_cls_feature(input_array, transform, model, device_id):

    assert len(input_array.shape) == 3  # 2D image

    """flipping the input if needed"""
    # input_array = np.swapaxes(input_array, 0,1)

    input_rgb_array = input_array[np.newaxis, :, :, :]

    input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
    input_tensor = transform(input_tensor)
    feature_array = model.forward_features(input_tensor.to(device=torch.device('cuda', device_id)))[
        'x_norm_clstoken'].detach().cpu().numpy()
    # "x_norm_clstoken": x_norm[:, 0],
    # "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
    # "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
    # "x_norm": x_norm,
    # "x_prenorm": x,
    # "masks": masks,
    del input_tensor

    return feature_array