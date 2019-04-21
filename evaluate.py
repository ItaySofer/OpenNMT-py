#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from datetime import datetime
import json
import os

from onmt.evaluate.evaluator import build_evaluator
from onmt.utils.misc import read_lines_string, concate_level

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from onmt.utils.logging import init_logger


def main(opt):
    logger = init_logger(opt.log_file)

    evaluator = build_evaluator(opt)

    for level in opt.levels:
        src_path = concate_level(opt.src, level)
        tgt_path = concate_level(opt.tgt, level)
        pred_path = concate_level(opt.pred, level)

        logger.info("Evaluating level %d." % level)
        evaluator.evaluate(
            src_path=src_path,
            tgt_path=tgt_path,
            pred_path=pred_path)

    src_unified_path, tgt_unified_path, pred_unified_path = postprocess(opt)
    logger.info("Evaluating unified levels")
    evaluator.evaluate(
        src_path=src_unified_path,
        tgt_path=tgt_unified_path,
        pred_path=pred_unified_path)


def postprocess(opt):
    src_unified = []
    tgt_unified = []
    pred_unified = []
    unified_translations = {}
    for level in opt.levels:
        src_lines = read_lines_string(concate_level(opt.src, level))
        tgt_lines = read_lines_string(concate_level(opt.tgt, level))
        pred_lines = read_lines_string(concate_level(opt.pred, level))

        src_unified = src_unified + src_lines
        tgt_unified = tgt_unified + tgt_lines
        pred_unified = pred_unified + pred_lines

        for (src_line, tgt_line, pred_line) in zip(src_lines, tgt_lines, pred_lines):
            level_pred_dict = {level: {'tgt': tgt_line, 'pred': pred_line}}
            unified_translations.setdefault(src_line, {}).update(level_pred_dict)

    unified_translations_sorted = dict(sorted(unified_translations.items(), key=lambda kv: len(kv[1]), reverse=True))  # sort by number of levels for source sentence

    write_to_file(opt, 'src_tgt_pred.unified', json.dumps(unified_translations_sorted))
    src_unified_path = write_to_file(opt, 'src.unified', "".join(src_unified))
    tgt_unified_path = write_to_file(opt, 'tgt.unified', "".join(tgt_unified))
    pred_unified_path = write_to_file(opt, 'pred.unified', "".join(pred_unified))

    return src_unified_path, tgt_unified_path, pred_unified_path


def write_to_file(opt, file_name, output_content):
    file_name = file_name
    output_path = os.path.join(opt.output, file_name)
    out_file = open(output_path, mode='w+', encoding='utf-8')
    out_file.write(output_content)

    return output_path


def _get_parser():
    parser = ArgumentParser(description='evaluate.py')

    opts.config_opts(parser)
    opts.general_opts(parser)
    opts.evaluate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    setattr(opt, 'datetime', datetime.now().strftime("%b-%d_%H-%M-%S"))
    main(opt)
