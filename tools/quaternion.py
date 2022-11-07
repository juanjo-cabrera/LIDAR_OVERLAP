#!/usr/bin/env python
# encoding: utf-8
"""
The Quaternion orientation class
@Authors: Arturo Gil
@Time: April 2021

"""
import numpy as np
from tools.conversions import euler2rot, euler2q, quaternion2rot, q2euler, rot2quaternion
from tools import euler
from tools import rotationmatrix


class Quaternion():
    def __init__(self, array):
        self.array = np.array(array)

    def R(self):
        return rotationmatrix.RotationMatrix(quaternion2rot(self.array))

    def Euler(self):
        """
        Convert Quaternion to Euler angles XYZ
        """
        return euler.Euler(q2euler(self.array))

    def Q(self):
        return self

    def __str__(self):
        return str(self.array)

    def toarray(self):
        return self.array

    def __add__(self, Q):
        return Quaternion(self.array + Q.toarray())

    def __sub__(self, Q):
        return Quaternion(self.array - Q.toarray())

    def dot(self, Q):
        """
        quaternion dot product
        """
        return np.dot(self.array, Q.toarray())

    def qconj(self):
         """
         quaternion conjugate
         """
         q0 = self.array[0]
         qv = -self.array[1:4]
         return Quaternion(np.hstack(([q0], qv)))

    def __mul__(self, Q):
        """
        quaternion product
        """
        if isinstance(Q, Quaternion):
            q1 = self.array
            q2 = Q.toarray()
            e0 = q1[0]*q2[0]-np.dot(q1[1:4], q2[1:4])
            e1 = np.dot(q1[0], q2[1:4]) + np.dot(q2[0], q1[1:4]) + np.cross(q1[1:4], q2[1:4])
            q = np.hstack(([e0], e1))
            return Quaternion(q)
        # this assumes that the rightmost element is a float
        else:
            q = self.array*Q
            return Quaternion(q)

    def __truediv__(self, Q):
        """
        Division of quaternion by constant
        """
        q = self.array / Q
        return Quaternion(q)