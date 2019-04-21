#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from datetime import datetime

from onmt.evaluate.evaluator import build_evaluator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def main(opt):

    evaluator = build_evaluator(opt)
    evaluator.evaluate()


def _get_parser():
    parser = ArgumentParser(description='evaluate.py')

    opts.config_opts(parser)
    opts.evaluate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    setattr(opt, 'datetime', datetime.now().strftime("%b-%d_%H-%M-%S"))
    main(opt)
