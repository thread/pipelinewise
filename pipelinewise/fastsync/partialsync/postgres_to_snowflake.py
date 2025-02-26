#!/usr/bin/env python3
import os

from datetime import datetime
from typing import Union
from argparse import Namespace

from pipelinewise.fastsync.commons.target_snowflake import FastSyncTargetSnowflake
from pipelinewise.fastsync.commons.tap_postgres import FastSyncTapPostgres
from pipelinewise.fastsync.postgres_to_snowflake import REQUIRED_CONFIG_KEYS, tap_type_to_target_type
from pipelinewise.fastsync.commons import utils as common_utils
from pipelinewise.fastsync.partialsync.utils import (
    upload_to_s3, update_state_file, parse_args_for_partial_sync, diff_source_target_columns, load_into_snowflake)
from pipelinewise.logger import Logger

LOGGER = Logger().get_logger(__name__)


# pylint: disable=too-many-locals
def partial_sync_table(args: Namespace) -> Union[bool, str]:
    """Partial sync table for Postgres to Snowflake"""
    snowflake = FastSyncTargetSnowflake(args.target, args.transform)
    tap_id = args.target.get('tap_id')
    dbname = args.tap.get('dbname')
    try:
        postgres = FastSyncTapPostgres(args.tap, tap_type_to_target_type)

        # Get bookmark - Binlog position or Incremental Key value
        postgres.open_connection()

        # Get column differences
        target_schema = common_utils.get_target_schema(args.target, args.table)
        table_dict = common_utils.tablename_to_dict(args.table)
        target_table = table_dict.get('table_name')

        target_sf = {
            'sf_object': snowflake,
            'schema': target_schema,
            'table': target_table,
            'temp': table_dict.get('temp_table_name')
        }

        snowflake_types = postgres.map_column_types_to_target(args.table)

        # making target table if not exists
        snowflake.create_table(
            target_schema=target_schema,
            table_name=target_table,
            columns=snowflake_types['columns'],
            primary_key=snowflake_types.get('primary_key'),
            is_temporary=False,
            sort_columns=False,
            allow_replace_table=False
        )

        source_columns = snowflake_types.get('columns', [])
        columns_diff = diff_source_target_columns(target_sf, source_columns=source_columns)

        bookmark = common_utils.get_bookmark_for_table(args.table, args.properties, postgres, dbname=dbname)

        where_clause_sql = f' WHERE {args.column} >= \'{args.start_value}\''
        if args.end_value:
            where_clause_sql += f' AND {args.column} <= \'{args.end_value}\''

        file_parts = postgres.export_source_table_data(args, tap_id, where_clause_sql)

        # mark partial data as deleted in the target
        snowflake.query(
            f'UPDATE {target_schema}."{target_table.upper()}"'
            f' SET _SDC_DELETEd_AT = CURRENT_TIMESTAMP(){where_clause_sql} AND _SDC_DELETED_AT IS NULL')

        # Creating temp table in Snowflake
        primary_keys = snowflake_types.get('primary_key')
        snowflake.create_schema(target_schema)
        snowflake.create_table(
            target_schema, args.table, source_columns, primary_keys, is_temporary=True
        )

        postgres.close_connection()

        size_bytes = sum([os.path.getsize(file_part) for file_part in file_parts])
        _, s3_key_pattern = upload_to_s3(snowflake, file_parts, args.temp_dir)

        load_into_snowflake(target_sf, args, columns_diff, primary_keys, s3_key_pattern, size_bytes, where_clause_sql)

        update_state_file(args, bookmark)

        return True
    except Exception as exc:
        LOGGER.exception(exc)
        return f'{args.table}: {exc}'


def main_impl():
    """Main sync logic"""
    args = parse_args_for_partial_sync(REQUIRED_CONFIG_KEYS)
    start_time = datetime.now()

    # Log start info
    LOGGER.info(
        '''
        -------------------------------------------------------
        STARTING PARTIAL SYNC
        -------------------------------------------------------
            Table selected to sync         : %s
            Column                         : %s
            Start value                    : %s
            End value                      : %s
        -------------------------------------------------------
        ''', args.table, args.column, args.start_value, args.end_value
    )

    sync_excs = partial_sync_table(args=args)
    if isinstance(sync_excs, bool):
        sync_excs = None

    # Log summary
    end_time = datetime.now()
    LOGGER.info(
        '''
        -------------------------------------------------------
        PARTIAL SYNC FINISHED - SUMMARY
        -------------------------------------------------------
            Table selected to sync         : %s
            Column                         : %s
            Start value                    : %s
            End value                      : %s
            Exceptions during table sync   : %s

            Runtime                        : %s
        -------------------------------------------------------
        ''', args.table, args.column, args.start_value, args.end_value, sync_excs, end_time - start_time
    )

    if sync_excs is not None:
        raise SystemExit(1)


def main():
    """Main entry point"""
    try:
        main_impl()
    except Exception as exc:
        LOGGER.critical(exc)
        raise exc
