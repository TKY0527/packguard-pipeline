"""Seven inline checkpoints. Day 1 — stub implementations that consume mock fixtures."""

from .c1_dicing import DicingCheckpoint
from .c2_die_attach import DieAttachCheckpoint
from .c3_wire_bond import WireBondCheckpoint
from .c4_molding import MoldingCheckpoint
from .c5_reflow import ReflowCheckpoint
from .c6_test import TestCheckpoint
from .c7_final_gate import FinalGateCheckpoint

ALL_CHECKPOINTS = [
    DicingCheckpoint(),
    DieAttachCheckpoint(),
    WireBondCheckpoint(),
    MoldingCheckpoint(),
    ReflowCheckpoint(),
    TestCheckpoint(),
    FinalGateCheckpoint(),
]

__all__ = [
    "ALL_CHECKPOINTS",
    "DicingCheckpoint",
    "DieAttachCheckpoint",
    "WireBondCheckpoint",
    "MoldingCheckpoint",
    "ReflowCheckpoint",
    "TestCheckpoint",
    "FinalGateCheckpoint",
]
