"""
Contextualized syntax: a communication act ("source S said ...") vs a factual record (an observation). The world model
was pretrained on web text wrapped as acts and on computed data as records, so it learned to believe records and to
treat acts as evidence about their sources.
"""


def act(text, **meta):
    """A communication act: the source described by meta (str values, empty ones dropped) said text."""
    header = " | ".join(f"{k}: {v}" for k, v in meta.items() if v)
    return f"<|act_start|>{header}<|said|>{text}<|act_end|>"


def fact(text):
    return f"<|fact_start|>{text}<|fact_end|>"
