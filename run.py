import os
import sys
from _ast import Lambda
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from shapely.affinity import affine_transform
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch import nn
from cytomine import CytomineJob
from cytomine.models import Job, ImageInstanceCollection, AttachedFileCollection, AnnotationCollection, Annotation
from sldc.locator import mask_to_objects_2d


from unet_model import UNet

MEAN = [0.78676176, 0.50835603, 0.78414893]
STD = [0.16071789, 0.24160224, 0.12767686]


def open_image(path):
    img = Image.open(path)
    trsfm = Compose([ToTensor(), Normalize(mean=MEAN, std=STD), Lambda(lambda x: x.unsqueeze(0))])
    return trsfm(img)


def predict_img(net, img_path, device, out_threshold=0.5):
    with torch.no_grad():
        x = open_image(img_path)
        logits = net(x.to(device))
        y_pred = nn.Softmax(dim=1)(logits)
        proba = y_pred.detach().cpu().squeeze(0).numpy()[1, :, :]
        return proba > out_threshold


def load_model(filepath):
    net = UNet(3, 2)
    net.cpu()
    net.load_state_dict(torch.load(filepath, map_location='cpu'))
    return net


class Monitor(object):
    def __init__(self, job, iterable, start=0, end=100, period=None, prefix=None):
        self._job = job
        self._start = start
        self._end = end
        self._update_period = period
        self._iterable = iterable
        self._prefix = prefix

    def update(self, *args, **kwargs):
        return self._job.job.update(*args, **kwargs)

    def _get_period(self, n_iter):
        """Return integer period given a maximum number of iteration """
        if self._update_period is None:
            return None
        if isinstance(self._update_period, float):
            return max(int(self._update_period * n_iter), 1)
        return self._update_period

    def _relative_progress(self, ratio):
        return int(self._start + (self._end - self._start) * ratio)

    def __iter__(self):
        total = len(self)
        for i, v in enumerate(self._iterable):
            period = self._get_period(total)
            if period is None or i % period == 0:
                statusComment = "{} ({}/{}).".format(self._prefix, i + 1, len(self))
                relative_progress = self._relative_progress(i / float(total))
                self._job.job.update(progress=relative_progress, statusComment=statusComment)
            yield v

    def __len__(self):
        return len(list(self._iterable))


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        working_path = str(Path.home())
        data_path = os.path.join(working_path, "data")
        model_path = os.path.join(working_path, "model")

        cj.job.update(status=Job.RUNNING, progress=0, statusComment="Download data...")

        # data download
        images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        filepaths = list()
        for img in images:
            filepath = os.path.join(data_path, img.originalFilename)
            img.download(filepath, override=False)
            filepaths.append(filepath)

        # 2. Call the image analysis workflow
        cj.job.update(progress=10, statusComment="Load model...")
        train_job = Job().fetch(cj.parameters.cytomine_id_job)
        attached_files = AttachedFileCollection(train_job).fetch()
        model_file = attached_files.find_by_attribute("filename", "model.pth")
        model_filepath = os.path.join(model_path, "model.joblib")
        model_file.download(model_filepath, override=True)
        net = load_model(model_filepath)
        device = torch.device("cpu")
        net.to(device)
        net.eval()

        annotations = AnnotationCollection()
        for image, filpath in Monitor(cj, zip(images, filepaths), start=20, end=75, period=0.05, prefix="Apply UNet to input images"):
            mask = predict_img(net, filepath, device=device, out_threshold=cj.parameters.threshold)
            objects = mask_to_objects_2d(mask)

            annotations.extend([
                Annotation(location=affine_transform(o, [1, 0, 0, -1, 0, image.height]),
                           id_image=image.id,
                           id_project=cj.parameters.cytomine_id_project)
                for o in objects
            ])

            del mask, objects

        # 4. Create and upload annotations
        cj.job.update(progress=70, statusComment="Uploading extracted annotation...")
        annotations.save()

        # 6. End the job
        cj.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])

