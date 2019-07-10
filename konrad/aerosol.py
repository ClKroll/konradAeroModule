import abc


class Aerosol(metaclass=abc.ABCMeta):
    def __init__(self, numlevels):
        self.optical_depth_at_55_micron
        self.optical_thickness_due_to_aerosol

    def update_aerosols(self, time):
        return