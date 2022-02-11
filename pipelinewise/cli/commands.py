"""
PipelineWise CLI - Commands
"""
import os
import shlex
import logging
import json
import time

from dataclasses import dataclass
from subprocess import PIPE, STDOUT, Popen
from pathlib import Path
from pathy import FluidPath
from typing import cast, Optional


from . import utils
from .errors import StreamBufferTooLargeException

LOGGER = logging.getLogger(__name__)
DEFAULT_STREAM_BUFFER_SIZE = 0  # Disabled by default
DEFAULT_STREAM_BUFFER_BIN = 'mbuffer'
MIN_STREAM_BUFFER_SIZE = 10
MAX_STREAM_BUFFER_SIZE = 2500

PARAMS_VALIDATION_RETRY_PERIOD_SEC = 2
PARAMS_VALIDATION_RETRY_TIMES = 3


def _verify_json_file(json_file_path: FluidPath, file_must_exists: bool, allowed_empty: bool) -> bool:
    """Checking if input file is a valid json or not, in some cases it is allowed to have an empty file,
     or it is allowed file not exists!
    """
    try:
        with json_file_path.open('r', encoding='utf-8') as json_file:
            json.load(json_file)
    except FileNotFoundError:
        return not file_must_exists
    except json.decoder.JSONDecodeError:
        if not allowed_empty or json_file_path.stat().st_size != 0:
            return False
    return True


def do_json_conf_validation(json_file: FluidPath, file_property: dict) -> bool:
    """
    Validating a json format config property and retry if it is invalid
    """
    for _ in range(PARAMS_VALIDATION_RETRY_TIMES):
        if _verify_json_file(json_file_path=json_file,
                             file_must_exists=file_property['file_must_exists'],
                             allowed_empty=file_property['allowed_empty']):
            return True

        time.sleep(PARAMS_VALIDATION_RETRY_PERIOD_SEC)
    return False


@dataclass
class TapParams:
    """
    TapParams validates json properties.
    """
    tap_id: str
    type: str
    bin: Path
    python_bin: Path
    config: FluidPath
    properties: FluidPath
    state: FluidPath

    def __post_init__(self):
        if not self.config:
            raise RunCommandException(
                f'Invalid json file for config: {self.config}')

        list_of_params_in_json_file = {
            'config': {'file_must_exists': True, 'allowed_empty': False},
            'properties': {'file_must_exists': True, 'allowed_empty': False},
            'state': {'file_must_exists': False, 'allowed_empty': True}
        }

        for param, file_property in list_of_params_in_json_file.items():
            valid_json = do_json_conf_validation(
                json_file=getattr(self, param, None),
                file_property=file_property
               ) if getattr(self, param, None) else True

            if not valid_json:
                raise RunCommandException(
                    f'Invalid json file for {param}: {getattr(self, param, None)}')


@dataclass
class TargetParams:
    """
    TargetParams validates json properties.
    """
    target_id: str
    type: str
    bin: Path
    python_bin: Path
    config: FluidPath

    def __post_init__(self):
        json_file = self.config

        valid_json = do_json_conf_validation(
            json_file=json_file,
            file_property={'file_must_exists': True, 'allowed_empty': False}) if json_file else False

        if not valid_json:
            raise RunCommandException(f'Invalid json file for config: {self.config}')


@dataclass
class TransformParams:
    """TransformParams."""
    bin: Path
    python_bin: Path
    config: FluidPath
    tap_id: str
    target_id: str


# pylint: disable=unnecessary-pass
class RunCommandException(Exception):
    """
    Custom exception to raise when run command fails
    """
    pass


def exists_and_executable(bin_path: Path) -> bool:
    """
    Checks if a given file exists and executable.
    It checks if the file exists and executable using the given
    absolute or relative path or via one of the path in the
    PATH environment variable

    Args:
         bin_path: Absolute or relative path
    Returns:
        boolean: True if file exists and executable, otherwise False

    """

    if not os.access(bin_path, os.X_OK):

        try:
            paths = f"{os.environ['PATH']}".split(':')
            (p for p in paths if os.access(Path(p, bin_path), os.X_OK)).__next__()
        except StopIteration:
            return False
    return True


def build_tap_command(
    tap: TapParams, profiling_mode: bool = False, profiling_dir: Optional[Path] = None
) -> str:
    """
    Builds a command that starts a singer tap connector with the
    required command line arguments

    Args:
        tap: TapParams instance with all the tap config details
        profiling_mode: Flag to indicate whether build the command with profiling
        profiling_dir: directory where profiling output should be dumped
    Returns:
        string of command line executable
    """
    # Following the singer spec the catalog JSON file needs to be passed by the --catalog argument
    # However some tap (i.e. tap-mysql and tap-postgres) requires it as --properties
    # This is probably for historical reasons and need to clarify on Singer slack channels
    catalog_argument = utils.get_tap_property_by_tap_type(
        tap.type, 'tap_catalog_argument'
    )

    state_arg = ''
    if tap.state and tap.state.is_file():
        state_arg = f'--state {utils.ensure_local(tap.state)}'

    tap_command = (
        f'{tap.bin} --config {utils.ensure_local(tap.config)} '
        f'{catalog_argument} {utils.ensure_local(tap.properties)} {state_arg}'
    )

    if profiling_mode:
        dump_file = utils.ensure_local(profiling_dir / f'tap_{tap.tap_id}.pstat')
        tap_command = f'{tap.python_bin} -m cProfile -o {dump_file} {tap_command}'

    return tap_command


def build_target_command(
    target: TargetParams, profiling_mode: bool = False, profiling_dir: Optional[Path] = None
) -> str:
    """
    Builds a command that starts a singer target connector with the
    required command line arguments

    Args:
        target: TargetParams instance with all the target config details
        profiling_mode: Flag to indicate whether build the command with profiling
        profiling_dir: directory where profiling output should be dumped
    Returns:
        string of command line executable
    """

    target_command = f'{target.bin} --config {utils.ensure_local(target.config)}'

    if profiling_mode:
        dump_file = utils.ensure_local(profiling_dir / f'target_{target.target_id}.pstat')
        target_command = (
            f'{target.python_bin} -m cProfile -o {dump_file} {target_command}'
        )

    return target_command


def build_transformation_command(
    transform: TransformParams, profiling_mode: bool = False, profiling_dir: Optional[Path] = None
) -> str:
    """
    Builds a command that starts a singer transformation connector
    with the required command line arguments

    Args:
        transform: TransformParams instance with all the transform config details
        profiling_mode: Flag to indicate whether build the command with profiling
        profiling_dir: directory where profiling output should be dumped
    Returns:
        string of command line executable if transformation found,
        None otherwise
    """
    trans_command = None

    # Detect if transformation is needed
    if transform.config.is_file():
        trans = utils.load_json(transform.config)
        if 'transformations' in trans and len(trans['transformations']) > 0:
            trans_command = f'{transform.bin} --config {utils.ensure_local(transform.config)}'

            if profiling_mode:
                dump_file = utils.ensure_local(
                    profiling_dir / f'transformation_{transform.tap_id}_{transform.target_id}.pstat'
                )

                trans_command = (
                    f'{transform.python_bin} -m cProfile -o {dump_file} {trans_command}'
                )

    return trans_command


def build_stream_buffer_command(
    buffer_size: int = 0,
    log_file: Optional[Path] = None,
    stream_buffer_bin: str = DEFAULT_STREAM_BUFFER_BIN,
) -> str:
    """
    Builds a command that buffers data between tap and target
    connectors to stream data asynchronously. Buffering streams
    avoids blocking taps to extract new data while targets
    is busy with loading the previous batch

    Args:
        buffer_size: Size of buffer in megabytes
        log_file: Log stream buffer status messages to this file
                           (Default is None, logging to stderr)
        stream_buffer_bin: binary executable of buffer implementation
                           (Default is mbuffer)

    Returns:
        string of command line executable
    Raises:
        StreamBufferTooLargeException if buffer size is greater than
            MAX_STREAM_BUFFER_SIZE
        StreamBufferBinaryNotFound if stream_buffer_bin binary executable
            not found
    """
    buffer_command = None

    if buffer_size and buffer_size > 0:
        # Buffer size cannot be less than min stream buffer size
        if buffer_size < MIN_STREAM_BUFFER_SIZE:
            buffer_size = MIN_STREAM_BUFFER_SIZE
        elif buffer_size > MAX_STREAM_BUFFER_SIZE:
            raise StreamBufferTooLargeException(buffer_size, MAX_STREAM_BUFFER_SIZE)

        buffer_command = f'{stream_buffer_bin} -m {buffer_size}M'

        # Log status to external file instead of stderr if log_file defined
        if log_file:
            log_file_stats = utils.log_file_with_status(log_file, utils.STATUS_RUNNING)
            buffer_command = f'{buffer_command} -q -l {log_file_stats}'

    return buffer_command


def build_singer_command(
    tap: TapParams,
    target: TargetParams,
    transform: TransformParams,
    stream_buffer_size: int = 0,
    stream_buffer_log_file: Optional[Path] = None,
    profiling_mode: bool = False,
    profiling_dir: Optional[Path] = None,
) -> str:
    """
    Builds a command that starts a full singer command with tap,
    target and optional transformation connectors. The connectors are
    connected by linux pipes, following the singer specification.

    Args:
        tap: NamedTuple with tap properties
        target: NamedTuple with target properties
        transform: NamedTuple with transform properties
        stream_buffer_size: in-memory buffer size between tap and target
        stream_buffer_log_file: Log stream buffer status messages to this file
                           (Default is None, logging to stderr)
        profiling_mode: Flag to indicate whether profiling is enabled or not
        profiling_dir: directory where profiling output should be dumped

    Returns:
        string of command line executable
    """
    tap_command = build_tap_command(tap, profiling_mode, profiling_dir)

    LOGGER.debug('Tap command: %s', tap_command)

    target_command = build_target_command(target, profiling_mode, profiling_dir)

    LOGGER.debug('Target command: %s', target_command)

    transformation_command = build_transformation_command(
        transform, profiling_mode, profiling_dir
    )
    LOGGER.debug('Transformation command: %s', transformation_command)

    stream_buffer_command = build_stream_buffer_command(
        stream_buffer_size, stream_buffer_log_file
    )

    LOGGER.debug('Buffer command: %s', stream_buffer_command)

    # Generate the final piped command with all the required components
    sub_commands = [
        tap_command,
        transformation_command,
        stream_buffer_command,
        target_command,
    ]
    command = ' | '.join(list(filter(None, sub_commands)))

    return command


# pylint: disable=too-many-arguments
def build_fastsync_command(
    tap: TapParams,
    target: TargetParams,
    transform: TransformParams,
    venv_dir: Path,
    temp_dir: Path,
    tables: str = None,
    profiling_mode: bool = False,
    profiling_dir: Optional[Path] = None,
    drop_pg_slot: bool = False,
) -> str:
    """
    Builds a command that starts fastsync from a given tap to a
    given target with optional transformations.

    Args:
        drop_pg_slot: flag for fastsync to indicate whether to drop or not PG replication slot
        profiling_dir: directory where profiling output should be dumped
        profiling_mode: Flag to indicate whether build the command with profiling
        tap: NamedTuple with tap properties
        target: NamedTuple with target properties
        transform: NamedTuple with transform properties
        venv_dir: path to virtual environment directory
        temp_dir: Temporary dir to generate export temp files
        tables: List of specific tables to fastsync
                (Default is None, to sync every table)

    Returns:
        string of command line executable
    """
    fastsync_bin = utils.get_fastsync_bin(venv_dir, tap.type, target.type)
    ppw_python_bin = utils.get_pipelinewise_python_bin(venv_dir)

    command_args = ' '.join(
        list(
            filter(
                None,
                [
                    f'--tap {utils.ensure_local(tap.config)}',
                    f'--properties {utils.ensure_local(tap.properties)}',
                    f'--state {utils.ensure_local(tap.state)}',
                    f'--target {utils.ensure_local(target.config)}',
                    f'--temp_dir {temp_dir}',
                    f'--transform {utils.ensure_local(transform.config)}'
                    if transform.config and transform.config.is_file()
                    else '',
                    f'--tables {tables}' if tables else '',
                    '--drop_pg_slot' if drop_pg_slot else '',
                ],
            )
        )
    )

    command = f'{fastsync_bin} {command_args}'

    if profiling_mode:
        dump_file = utils.ensure_local(
            profiling_dir / f'fastsync_{tap.tap_id}_{target.target_id}.pstat'
        )
        command = f'{ppw_python_bin} -m cProfile -o {dump_file} {command}'

    LOGGER.debug('FastSync command: %s', command)

    return command


# pylint: disable=too-many-locals
def run_command(command: str, log_file: Optional[Path] = None, line_callback: callable = None):
    """
    Runs a shell command with or without log file with STDOUT and STDERR

    Args:
        command: A unix command to run
        log_file: Write stdout and stderr to log file
        line_callback: function to call on each line on stdout and stderr
    """
    piped_command = f"/bin/bash -o pipefail -c '{command}'"
    LOGGER.debug('Running command %s', piped_command)

    # Logfile is needed: Continuously polling STDOUT and STDERR and writing into a log file
    # Once the command finished STDERR redirects to STDOUT and returns _only_ STDOUT
    if log_file is not None:
        LOGGER.info('Writing output into %s', log_file)

        # Create log dir if not exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Status embedded in the log file name
        log_file_running = utils.log_file_with_status(log_file, utils.STATUS_RUNNING)
        log_file_failed = utils.log_file_with_status(log_file, utils.STATUS_FAILED)
        log_file_success = utils.log_file_with_status(log_file, utils.STATUS_SUCCESS)

        # Create running log file to block any other processes.
        log_file_running.touch()

        # Start command
        with Popen(shlex.split(piped_command), stdout=PIPE, stderr=STDOUT) as proc:
            with log_file_running.open('a+', encoding='utf-8') as logfile:
                stdout = ''
                while True:
                    line = proc.stdout.readline()
                    if line:
                        decoded_line = line.decode('utf-8')

                        if line_callback is not None:
                            decoded_line = line_callback(decoded_line)

                        stdout += decoded_line

                        logfile.write(decoded_line)
                        logfile.flush()
                    if proc.poll() is not None:
                        break

        proc_rc = proc.poll()
        if proc_rc != 0:
            # Add failed status to the log file name
            log_file_running.rename(log_file_failed)

            # Raise run command exception
            errors = ''.join(utils.find_errors_in_log_file(log_file_failed))
            raise RunCommandException(
                f'Command failed. Return code: {proc_rc}\n'
                f'Error(s) found:\n{errors}\n'
                f'Full log: {log_file_failed}'
            )

        # Add success status to the log file name
        log_file_running.rename(log_file_success)

        return [proc_rc, stdout, None]

    # No logfile needed: STDOUT and STDERR returns in an array once the command finished
    with Popen(shlex.split(piped_command), stdout=PIPE, stderr=PIPE) as proc:
        proc_tuple = proc.communicate()
        proc_rc = proc.returncode
        stdout = proc_tuple[0].decode('utf-8')
        stderr = proc_tuple[1].decode('utf-8')

        if proc_rc != 0:
            LOGGER.error(stderr)

    return [proc_rc, stdout, stderr]
