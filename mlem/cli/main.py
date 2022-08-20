import logging
from collections import defaultdict
from functools import partial, wraps
from gettext import gettext
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import typer
from click import (
    Abort,
    ClickException,
    Command,
    HelpFormatter,
    Parameter,
    pass_context,
)
from click.exceptions import Exit
from pydantic import ValidationError
from typer import Context, Option, Typer
from typer.core import TyperCommand, TyperGroup

from mlem import LOCAL_CONFIG, version
from mlem.cli.utils import (
    NOT_SET,
    CallContext,
    _extract_examples,
    _format_validation_error,
    get_extra_keys,
)
from mlem.constants import MLEM_DIR, PREDICT_METHOD_NAME
from mlem.core.errors import MlemError
from mlem.telemetry import telemetry
from mlem.ui import EMOJI_FAIL, EMOJI_MLEM, bold, cli_echo, color, echo


class MlemFormatter(HelpFormatter):
    def write_heading(self, heading: str) -> None:
        super().write_heading(bold(heading))


class MlemMixin(Command):
    def __init__(
        self,
        *args,
        examples: Optional[str],
        section: str = "other",
        aliases: List[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.examples = examples
        self.section = section
        self.aliases = aliases
        self.rich_help_panel = section.capitalize()

    def collect_usage_pieces(self, ctx: Context) -> List[str]:
        return [p.lower() for p in super().collect_usage_pieces(ctx)]

    def get_help(self, ctx: Context) -> str:
        """Formats the help into a string and returns it.

        Calls :meth:`format_help` internally.
        """
        formatter = MlemFormatter(
            width=ctx.terminal_width, max_width=ctx.max_content_width
        )
        self.format_help(ctx, formatter)
        return formatter.getvalue().rstrip("\n")

    def format_epilog(self, ctx: Context, formatter: HelpFormatter) -> None:
        super().format_epilog(ctx, formatter)
        if self.examples:
            with formatter.section("Examples"):
                formatter.write(self.examples)


class MlemCommand(
    MlemMixin,
    TyperCommand,
):
    def __init__(
        self,
        name: Optional[str],
        section: str = "other",
        aliases: List[str] = None,
        help: Optional[str] = None,
        dynamic_options_generator: Callable[
            [CallContext], Iterable[Parameter]
        ] = None,
        dynamic_metavar: str = None,
        lazy_help: Optional[Callable[[], str]] = None,
        **kwargs,
    ):
        self.dynamic_metavar = dynamic_metavar
        self.dynamic_options_generator = dynamic_options_generator
        examples, help = _extract_examples(help)
        self._help = help
        self.lazy_help = lazy_help
        super().__init__(
            name=name,
            section=section,
            aliases=aliases,
            examples=examples,
            help=help,
            **kwargs,
        )

    def make_context(
        self,
        info_name: Optional[str],
        args: List[str],
        parent: Optional[Context] = None,
        **extra: Any,
    ) -> Context:
        args_copy = args[:]
        ctx = super().make_context(info_name, args, parent, **extra)
        if not self.dynamic_options_generator:
            return ctx
        extra_args = ctx.args
        while extra_args:
            with ctx.scope(cleanup=False):
                self.parse_args(ctx, args_copy)
            if ctx.args == extra_args:
                break
            extra_args = ctx.args

        return ctx

    def invoke(self, ctx: Context) -> Any:
        ctx.params = {k: v for k, v in ctx.params.items() if v != NOT_SET}
        return super().invoke(ctx)

    def get_params(self, ctx) -> List["Parameter"]:
        res: List[Parameter] = (
            list(
                self.dynamic_options_generator(
                    CallContext(ctx.params, get_extra_keys(ctx.args))
                )
            )
            if self.dynamic_options_generator is not None
            else []
        )
        res = res + super().get_params(ctx)
        if self.dynamic_metavar is not None:
            kw_param = [p for p in res if p.name == self.dynamic_metavar]
            if len(kw_param) > 0:
                res.remove(kw_param[0])
        return res

    @property
    def help(self):
        if self.lazy_help:
            return self.lazy_help()
        return self._help

    @help.setter
    def help(self, value):
        self._help = value


class MlemGroup(MlemMixin, TyperGroup):
    order = ["common", "object", "runtime", "other"]

    def __init__(
        self,
        name: Optional[str] = None,
        commands: Optional[
            Union[Dict[str, Command], Sequence[Command]]
        ] = None,
        section: str = "other",
        aliases: List[str] = None,
        help: str = None,
        **attrs: Any,
    ) -> None:
        examples, help = _extract_examples(help)
        super().__init__(
            name=name,
            help=help,
            examples=examples,
            aliases=aliases,
            section=section,
            commands=commands,
            **attrs,
        )

    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            # What is this, the tool lied about a command.  Ignore it
            if cmd is None:
                continue
            if cmd.hidden:
                continue

            commands.append((subcommand, cmd))

        # allow for 3 times the default spacing
        if len(commands) > 0:
            limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)

            sections = defaultdict(list)
            for subcommand, cmd in commands:
                help = cmd.get_short_help_str(limit)
                if isinstance(cmd, (MlemCommand, MlemGroup)):
                    section = cmd.section
                    aliases = (
                        f" ({','.join(cmd.aliases)})" if cmd.aliases else ""
                    )
                else:
                    section = "other"
                    aliases = ""

                sections[section].append((subcommand + aliases, help))

            for section in self.order:
                if sections[section]:
                    with formatter.section(
                        gettext(f"{section} commands".capitalize())
                    ):
                        formatter.write_dl(sections[section])

    def get_command(self, ctx: Context, cmd_name: str) -> Optional[Command]:
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if (
                isinstance(cmd, (MlemCommand, MlemGroup))
                and cmd.aliases
                and cmd_name in cmd.aliases
            ):
                return cmd
        return None


def mlem_group(section, aliases: Optional[List[str]] = None):
    class MlemGroupSection(MlemGroup):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, section=section, aliases=aliases, **kwargs)

    return MlemGroupSection


app = Typer(
    cls=MlemGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
# available from typer>=0.6
app.pretty_exceptions_enable = False
app.pretty_exceptions_show_locals = False


@app.callback(no_args_is_help=True, invoke_without_command=True)
def mlem_callback(
    ctx: Context,
    show_version: bool = Option(
        False, "--version", help="Show version and exit"
    ),
    verbose: bool = Option(
        False, "--verbose", "-v", help="Print debug messages"
    ),
    traceback: bool = Option(False, "--traceback", "--tb", hidden=True),
):
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialize any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment

    Examples:
        $ mlem init
        $ mlem list https://github.com/iterative/example-mlem
        $ mlem clone models/logreg --project https://github.com/iterative/example-mlem --rev main logreg
        $ mlem link logreg latest
        $ mlem apply latest https://github.com/iterative/example-mlem/data/test_x -o pred
        $ mlem serve latest fastapi -c port=8001
        $ mlem build latest docker_dir -c target=build/ -c server.type=fastapi
    """
    if ctx.invoked_subcommand is None and show_version:
        with cli_echo():
            echo(EMOJI_MLEM + f"MLEM Version: {version.__version__}")
    if verbose:
        logger = logging.getLogger("mlem")
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    ctx.obj = {"traceback": traceback or LOCAL_CONFIG.DEBUG}


def mlem_command(
    *args,
    section="other",
    aliases=None,
    options_metavar="[options]",
    parent=app,
    mlem_cls=None,
    dynamic_metavar=None,
    dynamic_options_generator=None,
    lazy_help=None,
    pass_ctx: bool = False,
    **kwargs,
):
    def decorator(f):
        if len(args) > 0:
            cmd_name = args[0]
        else:
            cmd_name = kwargs.get("name", f.__name__)
        context_settings = kwargs.get("context_settings", {})
        if dynamic_options_generator:
            context_settings.update(
                {"allow_extra_args": True, "ignore_unknown_options": True}
            )

        @parent.command(
            *args,
            options_metavar=options_metavar,
            context_settings=context_settings,
            **kwargs,
            cls=partial(
                mlem_cls or MlemCommand,
                section=section,
                aliases=aliases,
                dynamic_options_generator=dynamic_options_generator,
                dynamic_metavar=dynamic_metavar,
                lazy_help=lazy_help,
            ),
        )
        @wraps(f)
        @pass_context
        def inner(ctx, *iargs, **ikwargs):
            res = {}
            error = None
            try:
                if pass_ctx:
                    iargs = (ctx,) + iargs
                with cli_echo():
                    res = f(*iargs, **ikwargs) or {}
                res = {f"cmd_{cmd_name}_{k}": v for k, v in res.items()}
            except (ClickException, Exit, Abort) as e:
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                raise
            except MlemError as e:
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                if ctx.obj["traceback"]:
                    raise
                with cli_echo():
                    echo(EMOJI_FAIL + color(str(e), col=typer.colors.RED))
                raise typer.Exit(1)
            except ValidationError as e:
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                if ctx.obj["traceback"]:
                    raise
                msgs = "\n".join(_format_validation_error(e))
                with cli_echo():
                    echo(msgs)
                raise typer.Exit(1)
            except Exception as e:  # pylint: disable=broad-except
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                if ctx.obj["traceback"]:
                    raise
                with cli_echo():
                    echo(
                        EMOJI_FAIL
                        + color(
                            "Unexpected error: " + str(e), col=typer.colors.RED
                        )
                    )
                    echo(
                        "Please report it here: <https://github.com/iterative/mlem/issues>"
                    )
                raise typer.Exit(1)
            finally:
                telemetry.send_cli_call(cmd_name, error=error, **res)

        return inner

    return decorator


option_project = Option(
    None, "-p", "--project", help="Path to MLEM project", show_default="none"  # type: ignore
)
option_method = Option(
    PREDICT_METHOD_NAME,
    "-m",
    "--method",
    help="Which model method is to be applied",
)
option_rev = Option(None, "--rev", help="Repo revision to use", show_default="none")  # type: ignore
option_index = Option(
    None,
    "--index/--no-index",
    help="Whether to index output in .mlem directory",
)
option_external = Option(
    None,
    "--external",
    "-e",
    is_flag=True,
    help=f"Save result not in {MLEM_DIR}, but directly in project",
)
option_target_project = Option(
    None,
    "--target-project",
    "--tp",
    help="Project to save target to",
    show_default="none",  # type: ignore
)
option_json = Option(False, "--json", help="Output as json")
option_data_project = Option(
    None,
    "--data-project",
    "--dr",
    help="Project with data",
)
option_data_rev = Option(
    None,
    "--data-rev",
    help="Revision of data",
)


def option_load(type_: str = None):
    type_ = type_ + " " if type_ is not None else ""
    return Option(
        None, "-l", "--load", help=f"File to load {type_}config from"
    )


def option_conf(type_: str = None):
    type_ = f"for {type_} " if type_ is not None else ""
    return Option(
        None,
        "-c",
        "--conf",
        help=f"Options {type_}in format `field.name=value`",
    )


def option_file_conf(type_: str = None):
    type_ = f"for {type_} " if type_ is not None else ""
    return Option(
        None,
        "-f",
        "--file_conf",
        help=f"File with options {type_}in format `field.name=path_to_config`",
    )
