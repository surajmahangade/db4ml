#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/snapmgr.h"
#include "access/xact.h"
#include "pgstat.h"
#include "tcop/tcopprot.h"
#include "commands/dbcommands.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(db4ml_launch_training);
PG_FUNCTION_INFO_V1(db4ml_detect_outliers);
PGDLLEXPORT void db4ml_worker_main(Datum main_arg) pg_attribute_noreturn();

/* Worker Entry Point */
void db4ml_worker_main(Datum main_arg) {
    int32 job_id = DatumGetInt32(main_arg);
    char *dbname = MyBgworkerEntry->bgw_extra;
    
    pqsignal(SIGTERM, die);
    BackgroundWorkerUnblockSignals();

    BackgroundWorkerInitializeConnection(dbname, NULL, 0);

    SetCurrentStatementStartTimestamp();
    StartTransactionCommand();
    PushActiveSnapshot(GetTransactionSnapshot());

    if (SPI_connect() != SPI_OK_CONNECT) {
        elog(ERROR, "db4ml_worker: could not connect to SPI manager");
    }

    char sql_cmd[256];
    snprintf(sql_cmd, sizeof(sql_cmd), "SELECT db4ml.internal_run_job(%d)", job_id);

    int ret = SPI_execute(sql_cmd, false, 0);

    if (ret != SPI_OK_SELECT) {
        elog(WARNING, "db4ml_worker: Error calling internal_run_job (SPI Code %d)", ret);
    }

    SPI_finish();
    PopActiveSnapshot();
    CommitTransactionCommand();

    proc_exit(0);
}

/* User-facing function to launch training */
Datum db4ml_launch_training(PG_FUNCTION_ARGS) {
    text *sql_query_txt = PG_GETARG_TEXT_PP(0);
    text *target_col_txt = PG_GETARG_TEXT_PP(1);
    text *algo_name_txt = PG_GETARG_TEXT_PP(2); 
    
    char *sql_query = text_to_cstring(sql_query_txt);
    char *target_col = text_to_cstring(target_col_txt);
    char *algo_name = text_to_cstring(algo_name_txt); 
    
    int64 job_id = 0;

    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("Failed to connect to SPI")));
    }
    
    const char *cmd = "INSERT INTO db4ml.jobs(sql_query, target_column, algo_name) VALUES($1, $2, $3) RETURNING id";
    
    Oid argtypes[3] = { TEXTOID, TEXTOID, TEXTOID }; 
    Datum values[3] = { 
        CStringGetTextDatum(sql_query), 
        CStringGetTextDatum(target_col), 
        CStringGetTextDatum(algo_name) 
    }; 
    
    if (SPI_execute_with_args(cmd, 3, argtypes, values, NULL, false, 1) != SPI_OK_INSERT_RETURNING) { 
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to insert job into queue")));
    }
    
    bool isnull;
    job_id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    SPI_finish();

    /* Launch Background Worker */
    BackgroundWorker worker;
    MemSet(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = BGW_NEVER_RESTART;
    
    sprintf(worker.bgw_library_name, "db4ml");
    sprintf(worker.bgw_function_name, "db4ml_worker_main");
    snprintf(worker.bgw_name, BGW_MAXLEN, "db4ml worker job %ld", job_id);
    
    worker.bgw_main_arg = Int32GetDatum((int32)job_id);
    
    char *dbname = get_database_name(MyDatabaseId);
    if (dbname) {
        snprintf(worker.bgw_extra, BGW_EXTRALEN, "%s", dbname);
    } else {
        ereport(ERROR, (errmsg("Failed to get database name")));
    }

    BackgroundWorkerHandle *handle;
    if (!RegisterDynamicBackgroundWorker(&worker, &handle)) {
        ereport(ERROR, (errmsg("Failed to register dynamic background worker")));
    }

    PG_RETURN_INT64(job_id);
}

/* SQL-Integrated Outlier Detection Function */
Datum db4ml_detect_outliers(PG_FUNCTION_ARGS) {
    text *sql_query_txt = PG_GETARG_TEXT_PP(0);
    text *method_txt = PG_GETARG_TEXT_PP(1);
    
    char *sql_query = text_to_cstring(sql_query_txt);
    char *method = text_to_cstring(method_txt);
    
    int64 job_id = 0;

    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("Failed to connect to SPI")));
    }
    
    /* Insert outlier detection job */
    const char *cmd = "INSERT INTO db4ml.jobs(sql_query, target_column, algo_name) VALUES($1, NULL, $2) RETURNING id";
    
    Oid argtypes[2] = { TEXTOID, TEXTOID }; 
    char algo_name[128];
    snprintf(algo_name, sizeof(algo_name), "outlier_%s", method);
    
    Datum values[2] = { 
        CStringGetTextDatum(sql_query), 
        CStringGetTextDatum(algo_name)
    }; 
    
    if (SPI_execute_with_args(cmd, 2, argtypes, values, NULL, false, 1) != SPI_OK_INSERT_RETURNING) { 
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to insert outlier detection job")));
    }
    
    bool isnull;
    job_id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    SPI_finish();

    /* Launch Background Worker */
    BackgroundWorker worker;
    MemSet(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = BGW_NEVER_RESTART;
    
    sprintf(worker.bgw_library_name, "db4ml");
    sprintf(worker.bgw_function_name, "db4ml_worker_main");
    snprintf(worker.bgw_name, BGW_MAXLEN, "db4ml outlier detection job %ld", job_id);
    
    worker.bgw_main_arg = Int32GetDatum((int32)job_id);
    
    char *dbname = get_database_name(MyDatabaseId);
    if (dbname) {
        snprintf(worker.bgw_extra, BGW_EXTRALEN, "%s", dbname);
    } else {
        ereport(ERROR, (errmsg("Failed to get database name")));
    }

    BackgroundWorkerHandle *handle;
    if (!RegisterDynamicBackgroundWorker(&worker, &handle)) {
        ereport(ERROR, (errmsg("Failed to register dynamic background worker")));
    }

    PG_RETURN_INT64(job_id);
}