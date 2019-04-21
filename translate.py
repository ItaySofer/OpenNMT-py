#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import json
import os
from datetime import datetime

from onmt.utils.logging import init_logger
from onmt.utils.misc import read_lines_string, concate_level
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    for level in opt.levels:
        logger.info("Reading source and target files: %s %s. of level %s" % (opt.src, opt.tgt, level))

        src_path = concate_level(opt.src, level)
        tgt_path = concate_level(opt.tgt, level)

        logger.info("Translating level %d." % level)
        translator.translate(
            src=src_path,
            tgt=tgt_path,
            level=level,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
            )

    postprocess(opt, translator.out_files)


def postprocess(opt, out_files):
    src_unified = []
    tgt_unified = []
    pred_unified = []
    unified_translations = {}
    for level in opt.levels:
        src_lines = read_lines_string(concate_level(opt.src, level))
        tgt_lines = read_lines_string(concate_level(opt.tgt, level))
        pred_lines = get_pred_lines(out_files[level])

        src_unified = src_unified + src_lines
        tgt_unified = tgt_unified + tgt_lines
        pred_unified = pred_unified + pred_lines

        for (src_line, tgt_line, pred_line) in zip(src_lines, tgt_lines, pred_lines):
            level_pred_dict = {level: {'tgt': tgt_line, 'pred': pred_line}}
            unified_translations.setdefault(src_line, {}).update(level_pred_dict)

    unified_translations_sorted = dict(sorted(unified_translations.items(), key=lambda kv: len(kv[1]), reverse=True))  # sort by number of levels for source sentence

    write_to_file(opt, 'src_tgt_pred.unified.txt', json.dumps(unified_translations_sorted))
    write_to_file(opt, 'src.unified.txt', "".join(src_unified))
    write_to_file(opt, 'tgt.unified.txt', "".join(tgt_unified))
    write_to_file(opt, 'pred.unified.txt', "".join(pred_unified))


def get_pred_lines(pred_file):
    pred_file.seek(0)
    pred_lines = pred_file.readlines()
    return pred_lines


def write_to_file(opt, file_name, output):
    output_path = os.path.join(opt.output, opt.exp, opt.datetime)
    file_name = file_name
    out_file = open(os.path.join(output_path, file_name), mode='w+', encoding='utf-8')
    out_file.write(output)


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.general_opts(parser)
    opts.model_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    setattr(opt, 'datetime', datetime.now().strftime("%b-%d_%H-%M-%S"))
    main(opt)
