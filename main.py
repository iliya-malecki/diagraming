import click
import shapes
from shapes import (
    ExcalidrawClip,
    HasPoints,
    HasPressures,
    ElementType,
)
from protractor import resample_sequence, recognize, rotate
import numpy as np
import typing as t
import itertools

type Input = str | t.Literal["-"]


def get_data(input: Input):
    if input == "-":
        with click.get_text_stream("stdin") as stdin:
            return ExcalidrawClip.model_validate_json(stdin.read())
    else:
        return ExcalidrawClip.model_validate_json(input)


class Templates:
    def __init__(self):
        templates = []
        names = []
        rotations = []
        d: dict[ElementType, np.ndarray] = {
            "ellipse": np.c_[
                np.sin(np.linspace(-np.pi / 2, np.pi * 3 / 2)) + 1,
                np.cos(np.linspace(-np.pi / 2, np.pi * 3 / 2)),
            ],
            "rectangle": np.array(
                [
                    [0, 0],
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    [0, 0],
                ]
            ),
            "diamond": np.array(
                [
                    [0, 0],
                    [0.5, np.sqrt(0.75)],
                    [1, 0],
                    [0.5, -np.sqrt(0.75)],
                    [0, 0],
                ]
            ),
        }
        directions = lambda x: x, lambda x: x[::-1, ...]
        mirroring = lambda x: x, lambda x: x * np.array([1, -1])
        aspect_ratios = [lambda x: x * [1, n] for n in np.arange(0, 1, 0.2) + 0.2]
        for name, template in d.items():
            for transform_set in itertools.product(
                directions, mirroring, aspect_ratios
            ):
                for transformation in transform_set:
                    template = transformation(template)
                for angle in np.arange(0, 2 * np.pi, np.pi / 4):
                    templates.append(rotate(resample_sequence(template), angle))
                    names.append(name)
                    rotations.append(angle)

        self.vectors = np.stack(templates)
        self.names = np.array(names)
        self.rotations = np.array(rotations)


def masked[T](iterable: t.Iterable[T], mask: t.Iterable[bool]) -> t.Iterable[T]:
    for el, good in zip(iterable, mask):
        if good:
            yield el


def sift(masks: list[np.ndarray]):
    assert len(masks) > 0
    indecies = np.arange(len(masks[0]), dtype=np.int64)
    for mask in masks:
        indecies = indecies[mask]

    return indecies


@click.command()
@click.argument("input")
def excaliparse(input: Input):
    data = get_data(input)
    point_containing_mask = np.array([isinstance(x, HasPoints) for x in data.elements])
    points = np.stack(
        [resample_sequence(x.points) for x in data.elements if isinstance(x, HasPoints)]
    )
    scale = np.median(
        np.concatenate(
            [
                np.sqrt((np.diff(x.points, axis=1) ** 2).sum(axis=1))
                for x in data.elements
                if isinstance(x, HasPoints)
            ]
        )
    )
    sizes = (
        np.sqrt(((points.max(axis=1) - points.min(axis=1)) ** 2).sum(axis=-1)) / scale
    )
    for_recognition_mask = sizes > 5
    templates = Templates()
    template_ids, scores = recognize(
        points[for_recognition_mask, ...], templates.vectors
    )
    score_mask = scores > 0.95
    n: int
    template_name: ElementType
    for n, template_name, angle in zip(
        sift([point_containing_mask, for_recognition_mask, score_mask]),
        templates.names[template_ids[score_mask]],
        templates.rotations[template_ids[score_mask]],
    ):
        element = data.elements[n]
        assert isinstance(element, HasPoints)
        assert isinstance(element, HasPressures)
        # keep this boilerplate here while im not sure what specialcasing i need
        if template_name == "ellipse":
            data.elements[n] = shapes.Ellipse.from_base(element, angle=angle)
        elif template_name == "rectangle":
            data.elements[n] = shapes.Rectange.from_base(element, angle=angle).round(3)
        elif template_name == "diamond":
            data.elements[n] = shapes.Diamond.from_base(element, angle=angle).round(2)
        elif template_name == "arrow":
            raise NotImplementedError()
        elif template_name == "freedraw":
            raise NotImplementedError()
        elif template_name == "line":
            raise NotImplementedError()
        else:
            t.assert_never(template_name)

    print(data.model_dump_json())


if __name__ == "__main__":
    excaliparse()
