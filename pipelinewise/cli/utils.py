"""
PipelineWise CLI - Utilities
"""
import errno
import json
import logging
import os
import re
import secrets
import string
import sys
import tempfile
import warnings
import jsonschema
from pathy import FluidPath, Pathy
import yaml

from io import StringIO
from datetime import date, datetime
from jinja2 import Template
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import VaultLib, get_file_vault_secret, is_encrypted_file
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.parsing.yaml.objects import AnsibleMapping, AnsibleVaultEncryptedUnicode
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import tap_properties
from .errors import InvalidConfigException

LOGGER = logging.getLogger(__name__)

STATUS_RUNNING = 'running'
STATUS_FAILED = 'failed'
STATUS_SUCCESS = 'success'
STATUS_TERMINATED = 'terminated'


class AnsibleJSONEncoder(json.JSONEncoder):
    """
    Simple encoder class to deal with JSON encoding of Ansible internal types

    This is required to convert YAML files with vault encrypted inline values to
    singer JSON configuration files
    """

    # pylint: disable=method-hidden,assignment-from-no-return
    def default(self, o: Any) -> Any:
        if isinstance(o, AnsibleVaultEncryptedUnicode):
            # vault object - serialise the decrypted value as a string
            value = str(o)
        elif isinstance(o, Mapping):
            # hostvars and other objects
            value = dict(o)
        elif isinstance(o, (date, datetime)):
            # date object
            value = o.isoformat()
        elif isinstance(o, (Path, Pathy)):
            value = str(o)
        else:
            # use default encoder
            value = super().default(o)
        return value


def is_json(stringss: str) -> bool:
    """
    Detects if a string is a valid json or not
    """
    try:
        json.loads(stringss)
    except Exception:
        return False
    return True


def is_json_file(path: FluidPath) -> bool:
    """
    Detects if a file is a valid json file or not
    """
    try:
        path = Pathy.fluid(path)
        if path.is_file():
            with path.open(encoding='utf-8') as jsonfile:
                if json.load(jsonfile):
                    return True
        return False
    except Exception:
        return False


def load_json(path: FluidPath) -> Optional[Any]:
    """
    Deserialize JSON file to python object
    """
    try:
        path = Pathy.fluid(path)
        LOGGER.debug('Parsing file at %s', path)
        if path.is_file():
            with path.open(encoding='utf-8') as jsonfile:
                return json.load(jsonfile)
        else:
            LOGGER.debug('No file at %s', path)
            return None
    except Exception as exc:
        raise Exception(f'Error parsing {path} {exc}') from exc


def is_state_message(line: str) -> bool:
    """
    Detects if a string is a valid state message
    """
    try:
        json_object = json.loads(line)
        return 'bookmarks' in json_object
    except Exception:
        return False


def save_json(data: Any, path: FluidPath) -> None:
    """
    Serializes and saves any data structure to JSON files
    """
    try:
        path = Pathy.fluid(path)
        LOGGER.debug('Saving JSON %s', path)
        with path.open('w', encoding='utf-8') as jsonfile:
            json.dump(
                data, jsonfile, cls=AnsibleJSONEncoder, indent=4, sort_keys=True
            )
    except Exception as exc:
        raise Exception(f'Cannot save JSON {path} {exc}') from exc


def is_yaml(strings: str) -> bool:
    """
    Detects if a string is a valid yaml or not
    """
    try:
        yaml.safe_load(strings)
    except Exception:
        return False
    return True


def is_yaml_file(path: Path) -> bool:
    """
    Detects if a file is a valid yaml file or not
    """
    try:
        if path.is_file():
            with path.open(encoding='utf-8') as yamlfile:
                if yaml.safe_load(yamlfile):
                    return True
        return False
    except Exception:
        return False


def get_tap_target_names(yaml_dir: Path) -> Tuple[Set[Path], Set[Path]]:
    """Retrieves names of taps and targets inside yaml_dir.

    Args:
        yaml_dir (str): Path to the directory, which contains taps and targets files with .yml extension.

    Returns:
        (tap_yamls, target_yamls): tap_yamls is a list of names inside yaml_dir with "tap_*.y(a)ml" pattern.
                                   target_yamls is a list of names inside yaml_dir with "target_*.y(a)ml" pattern.
    """
    yamls = []
    for extension in ('*.yml', '*.yaml'):
        yamls.extend([yaml_file.name for yaml_file in yaml_dir.glob(extension) if yaml_file.is_file()])

    target_yamls = {file for file in yamls if file.startswith('target_')}
    tap_yamls = {file for file in yamls if file.startswith('tap_')}

    return tap_yamls, target_yamls


def load_yaml(yaml_file: Path, vault_secret: Optional[Path] = None) -> Dict:
    """
    Load a YAML file into a python dictionary.

    The YAML file can be fully encrypted by Ansible-Vault or can contain
    multiple inline Ansible-Vault encrypted values. Ansible Vault
    encryption is ideal to store passwords or encrypt the entire file
    with sensitive data if required.
    """
    vault = VaultLib()

    if vault_secret:
        secret_file = get_file_vault_secret(filename=vault_secret, loader=DataLoader())
        secret_file.load()
        vault.secrets = [('default', secret_file)]

    data = None
    if yaml_file.is_file():
        with yaml_file.open('r', encoding='utf-8') as stream:
            # Render environment variables using jinja templates
            contents = stream.read()
            template = Template(contents)
            stream = StringIO(template.render(env_var=os.environ))
            try:
                if is_encrypted_file(stream):
                    file_data = stream.read()
                    data = yaml.safe_load(vault.decrypt(file_data, None))
                else:
                    loader = AnsibleLoader(stream, None, vault.secrets)
                    try:
                        data = loader.get_single_data()
                    except Exception as exc:
                        raise Exception(
                            f'Error when loading YAML config at {yaml_file} {exc}'
                        ) from exc
                    finally:
                        loader.dispose()
            except yaml.YAMLError as exc:
                raise Exception(
                    f'Error when loading YAML config at {yaml_file} {exc}'
                ) from exc
    else:
        LOGGER.debug('No file at %s', yaml_file)

    if isinstance(data, AnsibleMapping):
        data = dict(data)

    return data


def vault_encrypt(plaintext: str, secret: Path) -> bytes:
    """
    Vault encrypt a piece of data.
    """
    try:
        vault = VaultLib()
        secret_file = get_file_vault_secret(filename=secret, loader=DataLoader())
        secret_file.load()
        vault.secrets = [('default', secret_file)]

        return vault.encrypt(plaintext)
    except AnsibleError as exc:
        LOGGER.critical('Cannot encrypt string: %s', exc)
        sys.exit(1)


def vault_format_ciphertext_yaml(b_ciphertext: bytes, indent: int = 10, name: Optional[str] = None) -> str:
    """
    Format a ciphertext to YAML compatible string
    """
    block_format_var_name = ''
    if name:
        block_format_var_name = '%s: ' % name

    block_format_header = '%s!vault |' % block_format_var_name
    lines = []
    vault_ciphertext = to_text(b_ciphertext)

    lines.append(block_format_header)
    for line in vault_ciphertext.splitlines():
        lines.append('%s%s' % (' ' * indent, line))

    yaml_ciphertext = '\n'.join(lines)
    return yaml_ciphertext


def load_schema(name: str) -> Dict:
    """
    Load a json schema
    """
    path = Path(__file__).parent / 'schemas' / f'{name}.json'
    schema = load_json(path)

    if not schema:
        LOGGER.critical('Cannot load schema at %s', path)
        sys.exit(1)

    assert isinstance(schema, dict), 'File does not contain a schema.'
    return schema


def get_sample_file_paths() -> List[Path]:
    """
    Get list of every available sample files (YAML, etc.) with absolute paths
    """
    samples_dir = Path(__file__).parent / 'samples'
    return search_files(
        samples_dir, patterns=['config.yml', '*.yml.sample', 'README.md'], abs_path=True
    )


def validate(instance, schema):
    """
    Validate an instance under a given json schema
    """
    try:
        # Serialise vault encrypted objects to string
        schema_safe_inst = json.loads(json.dumps(instance, cls=AnsibleJSONEncoder))
        jsonschema.validate(instance=schema_safe_inst, schema=schema)
    except jsonschema.exceptions.ValidationError as ex:
        raise InvalidConfigException(f'json object doesn\'t match schema {schema}') from ex


def delete_empty_keys(dictionary: Dict) -> Dict:
    """
    Deleting every key from a dictionary where the values are empty
    """
    return {k: v for k, v in dictionary.items() if v is not None}


def delete_keys_from_dict(dic, keys):
    """
    Delete specific keys from a nested dictionary
    """
    if not isinstance(dic, (dict, list)):
        return dic
    if isinstance(dic, list):
        return [v for v in (delete_keys_from_dict(v, keys) for v in dic) if v]
    # pylint: disable=C0325  # False positive on tuples
    return {
        k: v
        for k, v in ((k, delete_keys_from_dict(v, keys)) for k, v in dic.items())
        if k not in keys
    }


def silentremove(path: FluidPath) -> None:
    """
    Deleting file with no error message if the file not exists
    """
    LOGGER.debug('Removing file at %s', path)
    try:
        path.unlink()
    except OSError as exc:

        # errno.ENOENT = no such file or directory
        if exc.errno != errno.ENOENT:
            raise


def search_files(
    search_dir: FluidPath, patterns: Optional[List[str]] = None, sort: bool = False, abs_path: bool = False
) -> List[FluidPath]:
    """
    Searching files in a specific directory that match a pattern
    """
    if patterns is None:
        patterns = ['*']

    search_dir = Pathy.fluid(search_dir)
    if search_dir.is_dir():
        # Search files and sort if required
        p_files = []
        for pattern in patterns:
            p_files.extend(
                [file for file in search_dir.glob(pattern) if file.is_file()]
            )
        if sort:
            p_files.sort(key=lambda x: x.lstat().st_mtime, reverse=True)

    return [path if not abs_path else path.absolute() for path in p_files]


def extract_log_attributes(log_file: FluidPath) -> Dict[str, Any]:
    """
    Extracting common properties from a log file name
    """
    LOGGER.debug('Extracting attributes from log file %s', log_file)
    log_file = Pathy.fluid(log_file)
    target_id = 'unknown'
    tap_id = 'unknown'
    timestamp = datetime.utcfromtimestamp(0).isoformat()
    sync_engine = 'unknown'
    status = 'unknown'

    try:
        # Extract attributes from log file name
        log_attr = re.search(r'(.*)-(.*)-(.*)\.(.*)\.log\.(.*)', str(log_file))
        target_id = log_attr.group(1)
        tap_id = log_attr.group(2)
        timestamp = datetime.strptime(log_attr.group(3), '%Y%m%d_%H%M%S').isoformat()
        sync_engine = log_attr.group(4)
        status = log_attr.group(5)

    # Ignore exception when attributes cannot be extracted - Defaults will be used
    except Exception:
        pass

    # Return as a dictionary
    return {
        'filename': str(log_file),
        'target_id': target_id,
        'tap_id': tap_id,
        'timestamp': timestamp,
        'sync_engine': sync_engine,
        'status': status,
    }


def get_tap_property(tap: Dict, property_key: str, temp_dir: Optional[Path] = None) -> Any:
    """
    Get a tap specific property value
    """
    tap_props_inst = tap_properties.get_tap_properties(tap, temp_dir)
    tap_props = tap_props_inst.get(tap.get('type'), tap_props_inst.get('DEFAULT', {}))

    return tap_props.get(property_key)


def get_tap_property_by_tap_type(tap_type: str, property_key: str) -> Any:
    """
    Get a tap specific property value by a tap type.

    Some attributes cannot derived only by tap type. These
    properties might not be returned as expected.
    """
    tap_props_inst = tap_properties.get_tap_properties()
    tap_props = tap_props_inst.get(tap_type, tap_props_inst.get('DEFAULT', {}))

    return tap_props.get(property_key)


def get_tap_extra_config_keys(tap: Dict, temp_dir: Optional[Path] = None) -> Any:
    """
    Get tap extra config property
    """
    return get_tap_property(tap, 'tap_config_extras', temp_dir)


def get_tap_stream_id(tap: Dict, database_name: str, schema_name: str, table_name: str) -> str:
    """
    Generate tap_stream_id in the same format as a specific
    tap generating it. They are not consistent.

    Stream id is the string that tha tap's discovery mode puts
    into the properties.json file
    """
    pattern = get_tap_property(tap, 'tap_stream_id_pattern')

    return (
        pattern.replace('{{database_name}}', f'{database_name}')
        .replace('{{schema_name}}', f'{schema_name}')
        .replace('{{table_name}}', f'{table_name}')
    )


def get_tap_stream_name(tap: Dict, database_name: str, schema_name: str, table_name: str) -> str:
    """
    Generate tap_stream_name in the same format as a specific
    tap generating it. They are not consistent.

    Stream name is the string that the tap puts into the output
    singer messages
    """
    pattern = get_tap_property(tap, 'tap_stream_name_pattern')

    return (
        pattern.replace('{{database_name}}', f'{database_name}')
        .replace('{{schema_name}}', f'{schema_name}')
        .replace('{{table_name}}', f'{table_name}')
    )


def get_tap_default_replication_method(tap: Dict) -> str:
    """
    Get the default replication method for a tap
    """
    return get_tap_property(tap, 'default_replication_method')


def get_fastsync_bin(venv_dir: Path, tap_type: str, target_type: str) -> Path:
    """
    Get the absolute path of a fastsync executable
    """
    source = tap_type.replace('tap-', '')
    target = target_type.replace('target-', '')
    fastsync_name = f'{source}-to-{target}'

    return venv_dir / 'pipelinewise' / 'bin' / fastsync_name


def get_pipelinewise_python_bin(venv_dir: Path) -> str:
    """
    Get the absolute path of a PPW python executable
    Args:
        venv_dir: path to the ppw virtual env

    Returns: path to python executable
    """
    return venv_dir / 'pipelinewise' / 'bin' / 'python'


# pylint: disable=redefined-builtin
def create_temp_file(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[FluidPath] = None,
    text: Optional[str] = None,
) -> Path:
    """
    Create temp file with parent directories if not exists
    """
    dir = Pathy.fluid(dir) if dir is not None else dir
    if isinstance(dir, Path):
        dir.mkdir(parents=True, exist_ok=True)
    elif isinstance(dir, Pathy):
        raise ValueError(f'Temporary directory cannot be at remote: {dir}')

    _, temp_file = tempfile.mkstemp(suffix, prefix, dir, text)
    return Path(temp_file)


def find_errors_in_log_file(file: FluidPath, max_errors: int = 10, error_pattern: str = None) -> List[str]:
    """
    Find error lines in a log file

    Args:
        file: file to read
        max_errors: max number of errors to find
        error_pattern: Custom exception pattern

    Returns:
        List of error messages found in the file
    """
    file = Pathy.fluid(file)
    # List of known exception patterns in logs
    known_error_patterns = re.compile(
        # PPW error log patterns
        r'CRITICAL|'
        r'EXCEPTION|'
        r'ERROR|'
        # Basic tap and target connector exception patterns
        r'pymysql\.err|'
        r'psycopg2\.*Error|'
        r'snowflake\.connector\.errors|'
        r'botocore\.exceptions\.|'
        # Generic python exceptions
        r'\.[E|e]xception|'
        r'\.[E|e]rror'
    )

    # Use known error patterns by default
    if not error_pattern:
        error_pattern = re.compile(known_error_patterns)

    errors = []
    if file and file.is_file():
        with file.open(encoding='utf-8') as file_object:
            for line in file_object:
                if len(re.findall(error_pattern, line)) > 0:
                    errors.append(line)

                    # Seek to the end of the file, if max_errors found
                    if len(errors) >= max_errors:
                        file_object.seek(0, 2)

    return errors


def generate_random_string(length: int = 8) -> str:
    """
    Generate cryptographically secure random strings
    Uses best practice from Python doc https://docs.python.org/3/library/secrets.html#recipes-and-best-practices
    Args:
        length: length of the string to generate
    Returns: random string
    """

    if length < 1:
        raise Exception('Length must be at least 1!')

    if 0 < length < 8:
        warnings.warn('Length is too small! consider 8 or more characters')

    return ''.join(
        secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length)
    )


def log_file_with_status(log_file: FluidPath, status: str) -> FluidPath:
    """
    Adds an extension to a log file that represents the
    actual status of the tap

    Args:
        log_file: log file path without status extension
        status: a string that will be appended to the end of log file

    Returns:
        Path object pointing to log file with status extension
    """
    return log_file.with_suffix(log_file.suffix + '.' + status)


def ensure_local(path: FluidPath) -> Path:
    """This function checks whether a file is remote and if so downloads a local copy."""
    path = Pathy.fluid(path)
    if isinstance(path, Pathy):
        path = Pathy.to_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
