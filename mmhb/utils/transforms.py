"""
einops wrappers for torch transforms compatibility
"""

from einops import rearrange, repeat

class RearrangeTransform(object):
    """
    Wrapper for einops.rearrange to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, img):
        img = rearrange(img, self.pattern)
        return img

class RepeatTransform(object):
    """
    Wrapper for einops.repeat to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern, b):
        self.pattern = pattern
        self.b = b
    def __call__(self, img):
        img = repeat(img, self.pattern, b=self.b)
        return img