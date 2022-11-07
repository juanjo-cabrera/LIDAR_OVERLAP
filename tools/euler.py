#!/usr/bin/env python
# encoding: utf-8
"""
The orientation class
@Authors: Arturo Gil
@Time: April 2021

"""
import numpy as np
from tools.conversions import euler2rot, euler2q, quaternion2rot, q2euler, rot2quaternion
from tools import rotationmatrix
from tools import quaternion


class Euler():
    def __init__(self, abg):
        self.abg = np.array(abg)

    def R(self):
        return rotationmatrix.RotationMatrix(euler2rot(self.abg))

    def Q(self):
        return quaternion.Quaternion(euler2q(self.abg))
