from __future__ import annotations
from pydantic import BaseModel, Field, field_serializer
import typing as t
from numpydantic import NDArray, dtype
from abc import abstractmethod
import numpy as np


# A way to ensure isinstance checks are synced with this field
class HasPoints(BaseModel):
    points: NDArray[t.Literal["*, 2"], dtype.Number]  # type: ignore

    @field_serializer("points")
    def serialise_points(self, points: NDArray):
        return points.tolist()


class HasPressures(BaseModel):
    pressures: (
        NDArray[t.Literal["*"], dtype.Number] | NDArray[t.Literal["0"], dtype.Number]  # type: ignore
    )

    @field_serializer("pressures")
    def serialise_pressures(self, pressures: NDArray):
        return pressures.tolist()


class Element(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    angle: float
    strokeColor: str
    backgroundColor: str
    fillStyle: str
    strokeWidth: int
    strokeStyle: str
    roughness: int
    opacity: int
    groupIds: list[str]
    frameId: str | None
    index: str
    roundness: Roundness | None
    seed: int
    version: int
    versionNonce: int
    isDeleted: bool
    boundElements: list | None
    updated: int
    link: None
    locked: bool
    text: str | None = None

    @t.final
    def get_base_as_dict(self):
        return {k: getattr(self, k) for k in Element.model_fields}

    # mark it abstract for red squiggles, not for runtime ABC behavior
    @classmethod
    @abstractmethod
    def from_base(cls, other: Element, **kwargs) -> t.Self:
        # prefer explicit implementation over DRY but clever solutions
        raise NotImplementedError()

    def round(self, extent: int):
        new = type(self)(**{k: getattr(self, k) for k in type(self).model_fields})
        new.roundness = Roundness(type=extent)
        return new


type ElementType = t.Literal[
    "rectangle",
    "diamond",
    "ellipse",
    "arrow",
    "line",
    "freedraw",
]


class Roundness(BaseModel):
    type: int


class Rectange(Element):
    type: t.Literal["rectangle"]

    @classmethod
    def from_base(cls, other: Element, **kwargs) -> t.Self:
        return cls(type="rectangle", **{**other.get_base_as_dict(), **kwargs})


class Diamond(Element):
    type: t.Literal["diamond"]

    @classmethod
    def from_base(cls, other: Element, **kwargs) -> t.Self:
        return cls(type="diamond", **{**other.get_base_as_dict(), **kwargs})


class Ellipse(Element):
    type: t.Literal["ellipse"]

    @classmethod
    def from_base(cls, other: Element, **kwargs) -> t.Self:
        return cls(type="ellipse", **{**other.get_base_as_dict(), **kwargs})


class Arrow(Element, HasPoints):
    type: t.Literal["arrow"]
    lastCommittedPoint: None
    startBinding: None
    endBinding: None
    startArrowhead: None
    endArrowhead: t.Literal["arrow"]
    elbowed: bool

    @classmethod
    def from_base(cls, other: Element, **kwargs) -> t.Self:
        return cls(
            type="arrow",
            **other.get_base_as_dict(),
            lastCommittedPoint=None,
            startBinding=None,
            endBinding=None,
            startArrowhead=None,
            endArrowhead="arrow",
            elbowed=False,
            **kwargs,
        )


class Line(Element, HasPoints):
    type: t.Literal["line"]
    lastCommittedPoint: None
    startBinding: None
    endBinding: None
    startArrowhead: None
    endArrowhead: None

    @classmethod
    def from_base(cls, other: Element, **kwargs) -> t.Self:
        return cls(
            type="line",
            **other.get_base_as_dict(),
            lastCommittedPoint=None,
            startBinding=None,
            endBinding=None,
            startArrowhead=None,
            endArrowhead=None,
            **kwargs,
        )


class FreeDraw(Element, HasPoints, HasPressures):
    type: t.Literal["freedraw"]
    simulatePressure: bool
    lastCommittedPoint: tuple[float, float] | None

    @classmethod
    def from_base(cls, other: Element, *, points: NDArray, **kwargs) -> t.Self:
        return cls(
            type="freedraw",
            **other.get_base_as_dict(),
            lastCommittedPoint=None,
            pressures=np.array([]),
            points=points,
            **kwargs,
        )


ExcalidrawElement = t.Annotated[
    Rectange | Diamond | Ellipse | Arrow | Line | FreeDraw,
    Field(discriminator="type"),
]


class ExcalidrawClip(BaseModel):
    type: t.Literal["excalidraw/clipboard"]
    elements: list[ExcalidrawElement]
    files: dict


def _check_element_type(x: ExcalidrawElement, y: ElementType):
    """
    Statically check if i done did it and mistyped / forgot cases of ElementType.
    It looks so weird because i need to fight superflexible type inference of pyright
    """

    def _helper[T: t.LiteralString](x: T) -> t.Callable[[T], T]: ...

    _helper(y)(x.type)
    _helper(x.type)(y)
