# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from visual_genome import visual_genome


# Set up visual_genome_<split> using rpn mode
# for version in ['1.0', '1.2']:
for version in ['1.2']:
    for split in ['train', 'val', 'test']:
        name = 'vg_{}_{}'.format(version, split)
        __sets[name] = (lambda split=split, version=version:
                        visual_genome(split, version))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
