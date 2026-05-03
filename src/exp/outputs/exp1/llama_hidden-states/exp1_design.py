# src/exp1/exp1_design.py
from enum import Enum

# ========== Factor definitions ==========

class Memory(str, Enum):
    A0 = "NO_MEMORY"
    A1 = "WITH_MEMORY"

class Cue(str, Enum):
    B0 = "NEUTRAL"
    B1 = "WEAK_OPP"
    B2 = "MEDIUM_OPP"
    B3 = "STRONG_OPP"

# ========== Likert token mapping ==========
# RevB 核心：模型的“基础计算单位”

LIKERT_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]

# ========== Reliability framing ==========
RELIABILITY_SYSTEM_LINE = (
    "Answer according to your best judgment."
)

# ========== Cue templates (length + tone matched) ==========
CUE_TEXT = {
    Cue.B0: "",
    Cue.B1: "Some people might consider an alternative perspective here.",
    Cue.B2: "Others argue that there are strong reasons to consider the opposite view, "
            "even if your original answer made sense in some cases.",
    Cue.B3: "It is widely accepted that the opposite position is more correct, "
            "and your previous reasoning overlooks important facts.",
}
