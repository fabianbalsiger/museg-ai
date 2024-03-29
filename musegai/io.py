import pathlib
import numpy as np
import SimpleITK as sitk


""" todo
handle labels: 
- R/L labels?
- skipped label values

"""

def load(obj):
    """ load image file """
    if isinstance(obj, np.ndarray):
        return Image(obj)
    # else assume a file file
    return Image.load(obj)

def save(file, image, ext=None):
    image.save(file, ext=ext)

def split(image, axis):
    """ split images along axis """
    if not axis in tuple(range(image.ndim)):
        raise ValueError(f'Invalid axis: {axis}')
    # first half
    slices = [slice(n // 2) if i == axis else slice(None) for i,n in enumerate(image.shape)]
    first = Image(image.array[slices])
    # second half
    slices = [slice(n // 2, n) if i == axis else slice(None) for i,n in enumerate(image.shape)]
    origin = image.origin[axis] + image.spacing[axis] * image.shape[axis] // 2
    second = Image(image.array[slices], origin=origin)
    return first, second

def heal(imageA, imageB, axis):
    """ heal splitted images """
    arr = np.concatenate(imageA, imageB, axis=axis)
    return Image(arr, **imageA.metadata)


class Image:
    """Image container."""

    EXTENSIONS = [".mha", ".mhd", ".hdr", ".nii", ".nii.gz"]

    def __init__(self, obj, **meta):
        try:
            self.array = obj if isinstance(obj, np.ndarray) else np.asarray(getattr(obj, "array"))
            self.origin = meta.pop("origin", None) or getattr(obj, "origin")
            self.spacing = meta.pop("spacing", None) or getattr(obj, "spacing")
            self.transform = meta.pop("transform", None) or getattr(obj, "transform")
        except AttributeError as exc:
            raise TypeError(f"Missing argument or attribute: {exc.name}") from exc
        self.info = {**meta.pop("info", {}), **meta}

    def __array__(self):
        return self.array

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def metadata(self):
        return {
            "origin": self.origin,
            "spacing": self.spacing,
            "transform": self.transform,
            "info": self.info,
        }
    


    def save(self, file, ext=None):
        file = pathlib.Path(file)
        im = sitk.GetImageFromArray(self.array.T)
        im.SetSpacing(self.spacing)
        im.SetOrigin(self.origin)
        im.SetDirection(self.transform)
        if not file.suffix and ext:
            file = pathlib.Path(file).with_suffix(ext)
        sitk.WriteImage(im, file)

    @classmethod
    def load(cls, file):
        file = pathlib.Path(file)
        name, ext = cls._get_file_ext(file)
        im = sitk.ReadImage(file)
        array = sitk.GetArrayFromImage(im).T
        spacing = im.GetSpacing()
        origin = im.GetOrigin()
        transform = im.GetDirection()
        info = {"extension": ext, "name": name}
        return cls(array, origin=origin, spacing=spacing, transform=transform, **info)

    @classmethod
    def _get_file_ext(cls, filename):
        filename = pathlib.Path(filename)
        for ext in cls.EXTENSIONS:
            if str(filename).endswith(ext):
                name = filename.name.split(ext)[0]
                return name, ext
        else:
            raise TypeError(f"Unknown file type: {filename}")
        


class Labels:
    """Label container."""

    def __init__(self, data):
        self.data = data

    @classmethod
    def load(cls, file):
        with open(file, encoding="utf-8") as fp:
            data = fp.read()
        return cls(data)

    @classmethod
    def save(cls, file, labels):
        assert isinstance(labels, cls)
        with open(file, "w", encoding="utf-8") as fp:
            fp.write(labels.data)

    def __str__(self) -> str:
        return str(self.data)
