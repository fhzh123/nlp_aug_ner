import re
import os
import sys
import tqdm
import emoji
import logging
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR

def encoding_text(list_x, tokenizer, max_len):

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(r'<[^>]+>')

    def clean(x):
        x = pattern.sub(' ', x)
        x = x.strip()
        return x

    encoded_text_list = list_x.map(lambda x: tokenizer.encode(
        clean(str(x)),
        max_length=max_len,
        truncation=True
    ))
    return encoded_text_list

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)