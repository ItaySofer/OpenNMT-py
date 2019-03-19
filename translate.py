#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from datetime import datetime

from onmt.utils.logging import init_logger
from onmt.utils.misc import read_lines, concate_level
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
