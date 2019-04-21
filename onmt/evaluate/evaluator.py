#!/usr/bin/env python
""" Evaluator Class and builder """
from __future__ import print_function
import os


def build_evaluator(opt, logger=None):
    evaluator = Evaluator.from_opt(
        opt,
        logger=logger
    )

    return evaluator


class Evaluator(object):
    """Evaluates translations.

    Args:
        report_rouge (bool): Print/log Rouge metric.
        report_bleu (bool): Print/log Bleu metric.
        report_sari (bool): Print/log Sari metric.
        report_flesch_reading_ease (bool): Print/log flesch reading ease metric.
        report_flesch_kincaid_grade_level (bool): Print/log flesch kincaid grade level metric.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            report_rouge=False,
            report_bleu=False,
            report_sari=False,
            report_flesch_reading_ease=False,
            report_flesch_kincaid_grade_level=False,
            logger=None):
        self.report_rouge = report_rouge
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.report_sari = report_sari
        self.report_flesch_reading_ease = report_flesch_reading_ease
        self.report_flesch_kincaid_grade_level = report_flesch_kincaid_grade_level

        self.logger = logger

    @classmethod
    def from_opt(
            cls,
            opt,
            logger=None):
        """Alternate constructor.

        Args:
            opt (argparse.Namespace): Command line options.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        return cls(
            report_rouge=opt.report_rouge,
            report_bleu=opt.report_bleu,
            report_sari=opt.report_sari,
            report_flesch_reading_ease=opt.report_flesch_reading_ease,
            report_flesch_kincaid_grade_level=opt.report_flesch_kincaid_grade_level,
            logger=logger)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def evaluate(
            self,
            src_path,
            tgt_path,
            pred_path):
        """Evaluates content of src, tgt and pred.

        Args:

        Returns:

        """

        if self.report_rouge:
            msg = self._report_rouge(tgt_path, pred_path)
            self._log(msg)
        if self.report_bleu:
            msg = self._report_bleu(tgt_path, pred_path)
            self._log(msg)
        if self.report_sari:
            msg = self._report_sari(src_path, tgt_path, pred_path)
            self._log(msg)
        if self.report_flesch_reading_ease:
            msg = self._report_flesch_reading_ease(pred_path)
            self._log(msg)
        if self.report_flesch_kincaid_grade_level:
            msg = self._report_flesch_kincaid_grade_level(pred_path)
            self._log(msg)

    def _report_rouge(self, tgt_path, pred_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        msg = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN" % (path, tgt_path),
            shell=True, stdin=open(pred_path, "r")
        ).decode("utf-8").strip()
        return msg

    def _report_bleu(self, tgt_path, pred_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, os.path.abspath(tgt_path)),
            stdin=open(pred_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_sari(self, src_path, tgt_path, pred_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "python %s/tools/sari.py %s %s" % (
                base_dir, os.path.abspath(src_path), os.path.abspath(tgt_path)),
            stdin=open(pred_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_flesch_reading_ease(self, pred_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "python %s/tools/readability/readability.py \"Flesch Reading Ease\"" % base_dir,
            stdin=open(pred_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_flesch_kincaid_grade_level(self, pred_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "python %s/tools/readability/readability.py \"Flesch-Kincaid Grade Level\"" % base_dir,
            stdin=open(pred_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg
