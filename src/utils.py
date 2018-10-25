import logging
import json
from visualiser import VisdomWriter

logger = logging.getLogger(__name__)

def save_to_json(dict_item, fname):
    with open(fname, 'w') as f:
        json.dump(dict_item, f)
    
def save_to_txt(item, fname):
    with open(fname, 'a') as f:
        f.write('{}\n'.format(item))

class VisWriter:
    """Dummy Visdom Writer"""
    def __init__(self, vis=True):
        self.vis = vis
        if self.vis:
            self.writer = VisdomWriter(enabled=True, logger=logger)        

    def text(self, message, title):
        if self.vis:
            self.writer.text(message, title)

    def push(self, item, title):
        if self.vis:
            self.writer.push(item, title)