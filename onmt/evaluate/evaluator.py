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
        src_unified_path: Path to unified source sentences file
        tgt_unified_path: Path to unified target sentences file
        pred_unified_path: Path to unified predicted sentences file
        report_bleu (bool): Print/log Bleu metric.
        report_sari (bool): Print/log Sari metric.
        report_flesch_reading_ease (bool): Print/log flesch reading ease metric.
        report_flesch_kincaid_grade_level (bool): Print/log flesch kincaid grade level metric.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            src_unified_path,
            tgt_unified_path,
            pred_unified_path,
            report_bleu=False,
            report_rouge=False,
            report_sari=False,
            report_flesch_reading_ease=False,
            report_flesch_kincaid_grade_level=False,
            logger=None):
        self.src_unified_path = src_unified_path
        self.tgt_unified_path = tgt_unified_path
        self.pred_unified_path = pred_unified_path
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
            src_unified_path: Path to unified source sentences file
            tgt_unified_path: Path to unified target sentences file
            pred_unified_path: Path to unified predicted sentences file
            opt (argparse.Namespace): Command line options.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        return cls(
            src_unified_path=opt.src,
            tgt_unified_path=opt.tgt,
            pred_unified_path=opt.pred,
            report_bleu=opt.report_bleu,
            report_rouge=opt.report_rouge,
            report_sari=opt.report_sari,
            report_flesch_reading_ease=opt.report_flesch_reading_ease,
            report_flesch_kincaid_grade_level=opt.report_flesch_kincaid_grade_level,
            logger=logger)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def evaluate(self):
        """Evaluates content of src, tgt and pred.

        Args:

        Returns:

        """

        if self.report_bleu:
            msg = self._report_bleu()
            self._log(msg)
        if self.report_sari:
            msg = self._report_sari()
            self._log(msg)
        if self.report_flesch_reading_ease:
            msg = self._report_flesch_reading_ease()
            self._log(msg)
        # if self.report_flesch_kincaid_grade_level:
        #     msg = self._report_flesch_kincaid_grade_level()
        #     self._log(msg)

    def _report_bleu(self):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, os.path.abspath(self.tgt_unified_path)),
            stdin=open(self.pred_unified_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_sari(self):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "python %s/tools/sari.py %s %s" % (
                base_dir, os.path.abspath(self.src_unified_path), os.path.abspath(self.tgt_unified_path)),
            stdin=open(self.pred_unified_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_flesch_reading_ease(self):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")

        res = subprocess.check_output(
            "python %s/tools/readability/readability.py \"Flesch Reading Ease\"" % base_dir,
            stdin=open(self.pred_unified_path, "r"), shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    # def _report_flesch_kincaid_grade_level(self):
    #     import subprocess
    #     base_dir = os.path.abspath(__file__ + "/../../..")
    #     # Rollback pointer to the beginning.
    #     self.out_file.seek(0)
    #     print()
    #
    #     res = subprocess.check_output(
    #         "python %s/tools/readability/readability.py \"Flesch-Kincaid Grade Level\"" % base_dir,
    #         stdin=self.out_file, shell=True
    #     ).decode("utf-8")
    #
    #     msg = ">> " + res.strip()
    #     return msg
