"""Full-image and probability-blended tiled CUDA inference, one image/tile at a time."""
import numpy as np

from .images import normalized_image, axis_starts, merge_tiles


def forward_image(model, image, device, amp, probabilities=False):
    import torch
    import torch.nn.functional as F
    height, width = image.height, image.width
    x = torch.from_numpy(normalized_image(image)).unsqueeze(0).to(device)
    h, w = max(64, (height+31)//32*32), max(64, (width+31)//32*32)
    if (h, w) != (height, width):
        x = F.pad(x, (0, w-width, 0, h-height), mode='replicate')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        logits = model(x)[0, :, :height, :width]
    if probabilities:
        return logits.float().softmax(0).cpu().numpy()
    return logits.argmax(0).to(torch.uint8).cpu().numpy()


def predict_image(model, image, device, amp, mode='full', tile_size=512,
                  overlap=128, classes=2, threshold=None, save_probabilities=False):
    need_probabilities = mode == 'tile' or threshold is not None or save_probabilities
    if mode == 'full':
        output = forward_image(model, image, device, amp, need_probabilities)
        if not need_probabilities:
            return output, None
        probability = output
    elif mode == 'tile':
        def tiles():
            for y in axis_starts(image.height, tile_size, overlap):
                for x in axis_starts(image.width, tile_size, overlap):
                    crop = image.crop((x, y, min(x+tile_size, image.width), min(y+tile_size, image.height)))
                    yield y, x, forward_image(model, crop, device, amp, True)
        probability = merge_tiles(tiles(), classes, image.height, image.width)
    else:
        raise ValueError('Unknown inference mode')
    if threshold is not None:
        if classes != 2:
            raise ValueError('Custom foreground threshold requires binary model')
        labels = (probability[1] > threshold).astype(np.uint8)
    else:
        labels = probability.argmax(0).astype(np.uint8)
    return labels, probability if save_probabilities else None
