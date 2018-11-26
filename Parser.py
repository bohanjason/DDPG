import random
import numpy as np 
from hurry.filesize import size


POSTGRES_BYTES_SYSTEM = [
    (1024 ** 5, 'PB'),
    (1024 ** 4, 'TB'),
    (1024 ** 3, 'GB'),
    (1024 ** 2, 'MB'),
    (1024 ** 1, 'kB'),
    (1024 ** 0, 'B'),
]
class Parser(object):

    def rescaled(self, min_vals, max_vals, scaled_vals):
        min_vals = np.array(min_vals)
        max_vals = np.array(max_vals)
        scaled_vals = np.array(scaled_vals)
        return min_vals + scaled_vals * (max_vals - min_vals)

    def scaled(self, min_vals, max_vals, vals):
        min_vals = np.array(min_vals)
        max_vals = np.array(max_vals)
        vals = np.array(vals)
        return (vals - min_vals) * 1.0 / (max_vals - min_vals)

    def get_raw_size(self, value, system):
        for factor, suffix in system:
            if value.endswith(suffix):
                if len(value) == len(suffix):
                    amount = 1
                else:
                    try:
                        amount = int(value[:-len(suffix)])
                    except ValueError:
                        continue
                return amount * factor
        return None


    def convert_size(self, value):
        return  size(value, system=POSTGRES_BYTES_SYSTEM)

    def convert_int(self, value):
        return int(round(value))

    def get_knob_raw(self, value, knob_type):
        if knob_type == 'integer':
            return int(value)
        elif knob_type == 'size':
            return self.get_raw_size(value, POSTGRES_BYTES_SYSTEM)
        else:
            raise Exception('Knob Type does not support')

    def get_knob_readable(self, value, knob_type):
        if knob_type == 'integer':
            return self.convert_int(value)
        elif knob_type == 'size':
            value = int(round(value))
            return self.convert_size(value)
        else:
            raise Exception('Knob Type does not support')

