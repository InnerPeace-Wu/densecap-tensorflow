# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import numpy.random as npr
from six.moves import range
from lib.config import cfg
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from lib.fast_rcnn.nms_wrapper import nms
from lib.fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
    FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
    FONT = ImageFont.load_default()


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)

    return image


def draw_bounding_boxes(image, gt_boxes, im_info, phrases):

    num_boxes = gt_boxes.shape[0]
    gt_boxes_new = gt_boxes.copy()
    gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4].copy() / im_info[2])
    disp_image = Image.fromarray(np.uint8(image[0]))

    # show several(10) boxes for debugging
    show_ids = npr.choice(np.arange(num_boxes), size=5, replace=False)
    vocab_path = '%s/vocabulary.txt' % cfg.CACHE_DIR
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f]
    # vocab_extra = ['<EOS>', '<SOS>', '<PAD>']
    # for ex in vocab_extra:
    #     vocab.insert(0, ex)
    for idx, i in enumerate(show_ids):
        # this_class = int(gt_boxes_new[i, 4])
        # phrase = phrases[i] if len(phrases[i]) < cfg.TIME_STEPS else phrases[1:]
        # for adding gt bounding box
        if len(phrases[i]) < cfg.TIME_STEPS:
            phrase = phrases[i]
        # for adding predicted boxes
        else:
            phrase = []
            # phrases[i][1:] to remove the <SOS> token
            for p in phrases[i]:
                if p == cfg.END_INDEX:
                    break
                phrase.append(p)

        caption = ' '.join([vocab[j - 3] if j - 3 >= 0 else "" for j
                            in phrase])
        # caption = " ".join([vocab[j] for j in phrase[i])
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes_new[i, 0],
                                      gt_boxes_new[i, 1],
                                      gt_boxes_new[i, 2],
                                      gt_boxes_new[i, 3],
                                      '%s_%s' % (i, caption),
                                      FONT,
                                      color=STANDARD_COLORS[idx % NUM_COLORS])

    image[0, :] = np.array(disp_image)
    return image


def draw_densecap(image, scores, rois, im_info, cap_probs, bbox_pred):
    """
    bbox_pred: [None, 4]
    rois: [None, 5]

    """
    # for bbox unnormalization

    bbox_mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS).reshape((1, 4))
    bbox_stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS).reshape((1, 4))

    boxes = rois[:, 1:5] / im_info[2]
    # [None, 12]
    cap_ids = np.argmax(cap_probs, axis=1).reshape((-1, cfg.TIME_STEPS))

    # bbox target unnormalization
    box_deltas = bbox_pred * bbox_stds + bbox_mean

    # do the transformation
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, image.shape)

    pos_dets = np.hstack((pred_boxes, scores[:, 1][:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(pos_dets, cfg.TEST.NMS)
    pos_boxes = boxes[keep, :]
    cap_ids = cap_ids[keep, :]
    im_info[2] = 1.
    img_cap = draw_bounding_boxes(image, pos_boxes, im_info, cap_ids)

    return img_cap
