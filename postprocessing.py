import numpy as np
import torch
import os
import torch.nn.functional as F
from skimage.measure import label
from torchvision.transforms import ToPILImage, ToTensor, Resize
from torchvision.utils import save_image
from gan_mask_gen import pair_to_mask


def resize(x, target_shape):
    x = ToPILImage()(x.cpu().to(torch.float32))
    x = Resize(target_shape)(x)
    x = ToTensor()(x)
    return x.cuda()


def resize_min_edge(x, size):
    img_shape = x.shape[-2:]
    if img_shape[0] > img_shape[1]:
        x = resize((x[0] + 1.0) / 2.0, (size * img_shape[0] // img_shape[1], size))
    else:
        x = resize((x[0] + 1.0) / 2.0, (size, size * img_shape[1] // img_shape[0]))

    return x.unsqueeze(0) * 2.0 - 1.0


def connected_components_filter(*args):
    mask = args[-1].cpu().numpy()
    for i in range(len(mask)):
        component, num = label(mask[i], return_num=True, background=0)

        stats = np.zeros([num + 1])
        for comp in range(1, num + 1, 1):
            stats[comp] = np.sum(component == comp)

        max_component = np.argmax(stats)
        max_component_area = stats[max_component]

        mask[i] *= 0
        for comp in range(1, num + 1, 1):
            area = stats[comp]
            if float(area) / max_component_area > 0.2:
                mask[i][component == comp] = True

    return torch.from_numpy(mask).cuda()


class SegmentationInference(object):
    def __init__(self, model=None, resize_to=None):
        self.model = model
        self.resize_to = resize_to

    @torch.no_grad()
    def __call__(self, img, mask=None):
        return self.apply(img, mask=None)

    @torch.no_grad()
    def apply(self, img, mask=None):
        img_shape = img.shape[-2:]
        if mask is None:
            if self.model is not None:
                if self.resize_to is not None:
                    img = resize_min_edge(img, self.resize_to)
                mask = self.model(img)
            else:
                raise Exception(
                    'Eithr both (img, mask) should be provided or self.model is not None')

        if len(mask.shape) == 4:
            mask = (1.0 - torch.softmax(mask, dim=1))[:, 0]

        if self.resize_to is not None and mask.shape[-2:] != img_shape:
            mask = resize(mask, img_shape)

        return mask


class Threshold(SegmentationInference):
    def __init__(self, model=None, thr=0.5, resize_to=None):
        super(Threshold, self).__init__(model, resize_to)
        self.thr = thr

    @torch.no_grad()
    def __call__(self, img, mask=None):
        mask = self.apply(img, mask)
        return mask >= self.thr


class PseudoLabelGenerator:
    def __init__(self, G, bg_direction, zs,
                 mask_preprocessing=(),
                 mask_postprocessing=(connected_components_filter, ),
                 save_dir="results/cub_train_data"):
        self.G = G.cuda()
        self.G.eval()
        self.bg_direction = bg_direction.cuda()

        self.mask_preprocessing = mask_preprocessing
        self.mask_postprocessing = mask_postprocessing

        self.zs = zs
        self.idx = 0

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    @torch.no_grad()
    def make_noise(self):
        assert self.zs is not None
        if self.idx >= len(self.zs):
            raise StopIteration
        z = self.zs[self.idx].view(1, -1).cuda()
        self.idx += 1
        return z

    @torch.no_grad()
    def gen_samples(self, z=None, batch_size=None):
        assert (z is None) ^ (batch_size is None), 'one of: z, batch_size should be provided'

        if z is None:
            z = self.make_noise(batch_size)
        img = self.G(z)
        img_shifted_pos = self.G.gen_shifted(z, 5 * self.bg_direction.to(z.device))
        mask = pair_to_mask(img, img_shifted_pos)
        mask = self._apply_postproc(img, mask)

        return img, img_shifted_pos, mask

    @torch.no_grad()
    def _apply_preproc(self, img, intensity):
        for preproc in self.mask_preprocessing:
            intensity = preproc(img, intensity)
        return intensity

    @torch.no_grad()
    def _apply_postproc(self, img, mask):
        for postproc in self.mask_postprocessing:
            mask = postproc(img, mask)
        return mask

    @torch.no_grad()
    def __call__(self, img):
        z = self.make_noise()
        img_batch, img_pos_batch, ref_batch = self.gen_samples(z=z)

        # save it
        save_im = torch.cat([img, img_batch, img_pos_batch], dim=0).add_(1.0).mul_(0.5)
        mask_im = torch.stack([ref_batch, ref_batch, ref_batch], dim=1).float()
        save_im = torch.cat([save_im, mask_im], dim=0)
        save_image(save_im, os.path.join(self.save_dir, f"{self.idx:04d}.png"), nrow=1)

        return ref_batch
