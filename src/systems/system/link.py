from dataclasses import dataclass
from enum import Enum
from typing import Optional

import sympy as sp
from systems.system.types import Vec


@dataclass
class CoordinateT:
    x: Vec
    y: Vec


@dataclass
class LineT:
    start: CoordinateT
    end: CoordinateT


@dataclass
class StyledJointT(CoordinateT):
    edgecolor: str = 'k'
    facecolor: str = 'k'
    radius: float = 0.05
    zorder: float = 1.0


@dataclass
class StyledLinkT(LineT):
    linewidth: float = 1
    color: str = "black"
    zorder: float = 1.0


class LinkType(Enum):
    ROD = 0
    RING = 1
    DISK = 2
    CUSTOM = 3


class Link:
    """
    Represents a link in a kinematic chain.
    """

    def __init__(self, m, l, l_c, type: LinkType) -> None:
        """
        Args:
            m (float): mass of the link
            l (float): characteristic length
            type (LinkType): type of link
            I (float): used for a custom moment of inertia
        """

        self.m = m
        self.l = l
        self.l_c = l_c
        self.type = type
        self.I = Link.__construct_I(m, l, type)

    @staticmethod
    def __construct_I(m: sp.Symbol, l: sp.Symbol, type: LinkType) -> sp.Matrix:
        """
        Computes the moment of inertia of a link about its center of mass
        given its mass and length.
        """

        if type == LinkType.ROD:
            val = sp.Rational(1, 12) * m * l**2
        elif type == LinkType.RING:
            val = m * l**2
        elif type == LinkType.DISK:
            val = sp.Rational(1, 2) * m * l**2
        else:
            raise NotImplementedError("Moment of inertia for custom link type not implemented.")

        return sp.diag(val, val, val)
