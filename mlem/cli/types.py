from typing import Optional, Type

from pydantic import BaseModel
from typer import Argument

from mlem.cli.main import mlem_command
from mlem.cli.utils import iterate_type_fields, parse_type_field
from mlem.core.base import MlemABC, load_impl_ext
from mlem.core.objects import MlemObject
from mlem.ui import EMOJI_BASE, bold, color, echo
from mlem.utils.entrypoints import list_implementations


def type_fields_with_collection_examples(cls):
    for field in iterate_type_fields(cls):
        yield field
        if field.is_list or field.is_mapping:
            key = ".key" if field.is_mapping else ".0"
            yield from parse_type_field(
                field.path + key,
                field.type_,
                field.help,
                False,
                False,
                False,
                None,
            )


def explain_type(cls: Type[BaseModel]):
    echo(
        color("Type ", "")
        + color(cls.__module__ + ".", "yellow")
        + color(cls.__name__, "green")
    )
    if issubclass(cls, MlemABC):
        echo(color("MlemABC parent type: ", "") + color(cls.abs_name, "green"))
        echo(color("MlemABC type: ", "") + color(cls.__get_alias__(), "green"))
    if issubclass(cls, MlemObject):
        echo(
            color("MlemObject type name: ", "")
            + color(cls.object_type, "green")
        )
    echo((cls.__doc__ or "Class docstring missing").strip())
    fields = list(type_fields_with_collection_examples(cls))
    if not fields:
        echo("No fields")
    else:
        echo("Fields:")
    for field in fields:
        echo(field.to_text())


@mlem_command("types", hidden=True)
def list_types(
    abc: Optional[str] = Argument(
        None,
        help="Subtype to list implementations. List subtypes if not provided",
    ),
    sub_type: Optional[str] = Argument(None, help="Type of `meta` subtype"),
):
    """List MLEM types implementations available in current env.
    If subtype is not provided, list ABCs

    Examples:
        List ABCs
        $ mlem types

        List available server implementations
        $ mlem types server
    """
    if abc is None:
        for at in MlemABC.abs_types.values():
            echo(EMOJI_BASE + bold(at.abs_name) + ":")
            echo(
                f"\tBase class: {at.__module__}.{at.__name__}\n\t{(at.__doc__ or 'Class docstring missing').strip()}"
            )
    elif abc == MlemObject.abs_name:
        if sub_type is None:
            echo("\n".join(MlemObject.non_abstract_subtypes().keys()))
        else:
            mlem_object_type = MlemObject.non_abstract_subtypes()[sub_type]
            if mlem_object_type.__is_root__:
                echo(
                    "\n".join(
                        list_implementations(
                            MlemObject, mlem_object_type, include_hidden=False
                        )
                    )
                )
            else:
                explain_type(mlem_object_type)
    else:
        if sub_type is None:
            echo("\n".join(list_implementations(abc, include_hidden=False)))
        else:
            cls = load_impl_ext(abc, sub_type, True)
            explain_type(cls)
