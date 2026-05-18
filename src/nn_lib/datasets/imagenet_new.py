import os
from PIL import Image
from torchvision.datasets import ImageFolder
from ffcv.loader import Loader, OrderOption
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, IntDecoder
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
import numpy as np

def load_image(path):
    image = Image.open(path).convert('RGB')  

    return image

mean = np.array([0.485, 0.456, 0.406]) * 255
std = np.array([0.229, 0.224, 0.225]) * 255

class ImageNet():
    _default_shape = (3, 224, 224)

    def __new__(cls, 
                train: bool = True, 
                data_dir: str = None, 
                ffcv_format_save_dir: str = None,
                batch_size: int = 256,
                num_workers: int = 4,
                order_option: OrderOption = OrderOption.RANDOM,
                pipeline_image: list = [
                    RandomResizedCropRGBImageDecoder((224, 224), (0.08, 1.0)), 
                    ToTensor(), 
                    ToDevice('cuda', non_blocking = True),
                    ToTorchImage(),
                    NormalizeImage(mean, std, np.float16),
                ],
                pipeline_label: list = [
                    IntDecoder(),
                    ToTensor(), 
                    ToDevice('cuda', non_blocking = True)
                ],
                os_cache: bool = True,
                seed: int = 42,
                ):

        instance = super().__new__(cls)

        instance.train = train
        instance.data_dir = data_dir
        instance.ffcv_format_file = os.path.join(ffcv_format_save_dir if ffcv_format_save_dir is not None else data_dir, 
                                                 "imagenet_train.beton" if instance.train else "imagenet_val.beton")

        # ffcv settings.
        instance.batch_size = batch_size
        instance.num_workers = num_workers
        instance.order_option = order_option
        instance.pipeline_image = pipeline_image
        instance.pipeline_label = pipeline_label
        instance.os_cache = os_cache
        instance.seed = seed

        # Check if the FFCV format already exists; if not, convert the dataset to FFCV format and save it to disk.
        if not os.path.exists(instance.ffcv_format_file):
            instance.convert_to_ffcv_format()

        # Create an FFCV data loader that reads from the FFCV format dataset on disk.
        loader = Loader(
            instance.ffcv_format_file,
            batch_size = instance.batch_size,
            num_workers = instance.num_workers,
            pipelines = {
                'image': instance.pipeline_image,
                'label': instance.pipeline_label
            },
            order = instance.order_option,
        )

        return loader

    def convert_to_ffcv_format(self):
        # We need an indexable dataset for FFCV. 
        indexable_dataset = ImageFolder(
            root = os.path.join(self.data_dir, "train" if self.train else "val"),
            loader = load_image
        ) 

        # Create a DatasetWriter to convert the indexable dataset to FFCV format and save it to disk.
        writer = DatasetWriter(
            self.ffcv_format_file,
            {
                'image': RGBImageField(),
                'label': IntField()
            },
            num_workers = self.num_workers,
        )

        # Use the DatasetWriter to convert the indexable dataset to FFCV format and save it to disk. 
        writer.from_indexed_dataset(indexable_dataset)