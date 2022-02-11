"""
PipelineWise CLI
"""
import argparse
import os
import sys
import copy
import logging

from cProfile import Profile
from datetime import datetime
from typing import Optional, Tuple
from pkg_resources import get_distribution
from pathlib import Path
from pathy import FluidPath, Pathy

from .utils import generate_random_string
from .pipelinewise import PipelineWise
from ..logger import Logger

__version__ = get_distribution('pipelinewise').version
USER_HOME = Path('~').expanduser()
DEFAULT_CONFIG_DIR = USER_HOME / '.pipelinewise'
CONFIG_DIR = Pathy(os.environ.get('PIPELINEWISE_CONFIG_DIRECTORY', DEFAULT_CONFIG_DIR))
PROFILING_DIR = CONFIG_DIR / 'profiling'
PIPELINEWISE_DEFAULT_HOME = USER_HOME / 'pipelinewise'
PIPELINEWISE_HOME = Path(
    os.environ.setdefault('PIPELINEWISE_HOME', str(PIPELINEWISE_DEFAULT_HOME))
)
VENV_DIR = PIPELINEWISE_HOME / '.virtualenvs'
COMMANDS = [
    'init',
    'run_tap',
    'stop_tap',
    'discover_tap',
    'status',
    'test_tap_connection',
    'sync_tables',
    'import',
    'import_config',  # This is for backward compatibility; use 'import' instead
    'validate',
    'encrypt_string',
]


def __init_logger(log_file: Optional[Path] = None, debug: bool = False) -> Logger:
    """
    Initialise logger and update its handlers and level accordingly
    """
    # get logger for pipelinewise
    logger = Logger(debug).get_logger('pipelinewise')

    # copy log configuration: level and formatter
    level = logger.level
    formatter = copy.deepcopy(logger.handlers[0].formatter)

    # Create log file handler if required
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def __init_profiler(
    profiler_arg: bool, logger: logging.Logger
) -> Tuple[Optional[Profile], Optional[str]]:
    """
    Initialise profiling environment by creating a cprofile.Profiler instance, a folder where pstats can be dumped
    Args:
        profiler_arg: the value of profiler argument passed when running the command
        logger: a logger instance

    Returns:
        If profiling enabled, a tuple of profiler instance and profiling directory where the stats files
        would be dumped, otherwise, a tuple of nulls
    """
    if profiler_arg:
        logger.info('Profiling mode enabled')

        logger.debug('Creating & enabling profiler ...')

        profiler = Profile()
        profiler.enable()

        logger.debug('Profiler created.')

        profiling_dir = (
            PROFILING_DIR /
            f'{datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")}_{generate_random_string(10)}'
        )

        profiling_dir.mkdir(parents=True, exist_ok=True)
        logger.debug('Profiling directory "%s" created', profiling_dir)

        return profiler, profiling_dir

    logger.info('Profiling mode not enabled')

    return None, None


def __disable_profiler(
    profiler: Optional[Profile],
    profiling_dir: Optional[FluidPath],
    pstat_filename: Optional[str],
    logger: logging.Logger,
):
    """
    Disable given profiler and dump pipelinewise stats into a pStat file
    Args:
        profiler: optional instance of cprofile.Profiler to disable
        profiling_dir: profiling dir where pstat file will be created
        pstat_filename: custom pstats file name, the extension .pstat will be appended to the name
        logger: Logger instance to do some info and debug logging
    """
    if profiler is not None:
        logger.debug('disabling profiler and dumping stats...')

        profiler.disable()

        if not pstat_filename.endswith('.pstat'):
            pstat_filename = f'{pstat_filename}.pstat'

        dump_file = utils.ensure_local(profiling_dir) / pstat_filename

        logger.debug('Attempting to dump profiling stats in file "%s" ...', dump_file)
        profiler.dump_stats(dump_file)
        logger.debug('Profiling stats dump successful')

        dest = profiling_dir / pstat_filename
        dest.touch()
        dest.write_bytes(dump_file.read_bytes())
        logger.info('Profiling stats files are in folder "%s"', profiling_dir)

        profiler.clear()


# pylint: disable=too-many-branches,too-many-statements
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PipelineWise {} - Command Line Interface'.format(__version__),
        add_help=True,
    )
    parser.add_argument('command', type=str, choices=COMMANDS)
    parser.add_argument('--target', type=str, help='"Name of the target')
    parser.add_argument('--tap', type=str, help='Name of the tap')
    parser.add_argument('--tables', type=str, help='List of tables to sync')
    parser.add_argument(
        '--dir', type=Path, help='Path to directory with config'
    )
    parser.add_argument('--name', type=str, help='Name of the project')
    parser.add_argument('--secret', type=Path, help='Path to vault password file')
    parser.add_argument('--string', type=str)
    parser.add_argument(
        '--version',
        action='version',
        help='Displays the installed versions',
        version='PipelineWise {} - Command Line Interface'.format(__version__),
    )
    parser.add_argument('--log', type=Path, help='File to log into')
    parser.add_argument(
        '--extra_log',
        default=False,
        required=False,
        help='Copy singer and fastsync logging into PipelineWise logger',
        action='store_true',
    )
    parser.add_argument(
        '--debug',
        default=False,
        required=False,
        help='Forces the debug mode with logging on stdout and log level debug',
        action='store_true',
    )
    parser.add_argument(
        '--profiler',
        '-p',
        default=False,
        required=False,
        help='Enables code profiling mode using Python builtin profiler cProfile. '
        'The stats will be dumped into a folder in .pipelinewise/profiling',
        action='store_true',
    )

    args = parser.parse_args()

    # Command specific argument validations
    if args.command == 'init' and args.name is None:
        print('You must specify a project name using the argument --name')
        sys.exit(1)

    if args.command in ['discover_tap', 'test_tap_connection', 'run_tap', 'stop_tap']:
        if args.tap is None:
            print('You must specify a source name using the argument --tap')
            sys.exit(1)
        if args.target is None:
            print('You must specify a destination name using the argument --target')
            sys.exit(1)

    if args.command == 'sync_tables':
        if args.tap is None:
            print('You must specify a source name using the argument --tap')
            sys.exit(1)
        if args.target is None:
            print('You must specify a destination name using the argument --target')
            sys.exit(1)

    # import and import_config commands are synonyms
    #
    # import        : short CLI command name to import project
    # import_config : this is for backward compatibility; use 'import' instead from CLI
    if args.command == 'import' or args.command == 'import_config':
        if args.dir is None:
            print(
                'You must specify a directory path with config YAML files using the argument --dir'
            )
            sys.exit(1)

        # Every command argument is mapped to a python function with the same name, but 'import' is a
        # python keyword and can't be used as function name
        args.command = 'import_project'

    if args.command == 'validate' and args.dir is None:
        print(
            'You must specify a directory path with config YAML files using the argument --dir'
        )
        sys.exit(1)

    if args.command == 'encrypt_string':
        if not args.secret:
            print(
                'You must specify a path to a file with vault secret using the argument --secret'
            )
            sys.exit(1)
        if not args.string:
            print('You must specify a string to encrypt using the argument --string')
            sys.exit(1)

    logger = __init_logger(args.log, args.debug)

    profiler, profiling_dir = __init_profiler(args.profiler, logger)

    ppw_instance = PipelineWise(args, CONFIG_DIR, VENV_DIR, profiling_dir)

    try:
        getattr(ppw_instance, args.command)()
    finally:
        __disable_profiler(
            profiler, profiling_dir, f'pipelinewise_{args.command}', logger
        )


if __name__ == '__main__':
    main()
