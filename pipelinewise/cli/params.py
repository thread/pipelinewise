import json
import time

from dataclasses import dataclass
from pathlib import Path
from pathy import FluidPath

from .errors import RunCommandException

PARAMS_VALIDATION_RETRY_PERIOD_SEC = 2
PARAMS_VALIDATION_RETRY_TIMES = 3


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
        valid_json = do_json_conf_validation(
            json_file=self.config,
            file_property={'file_must_exists': True, 'allowed_empty': False}) if self.config else False

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
