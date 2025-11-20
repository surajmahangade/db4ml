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
// void db4ml_worker_main(Datum main_arg) pg_attribute_noreturn();
PGDLLEXPORT void db4ml_worker_main(Datum main_arg) pg_attribute_noreturn();

/* * WORKER ENTRY POINT
 * This is the process that runs in background.
 */
void db4ml_worker_main(Datum main_arg) {
    int32 job_id = DatumGetInt32(main_arg);
    char *dbname = MyBgworkerEntry->bgw_extra; /* Database name passed in extra */
    
    /* Establish signal handlers (standard boilerplate) */
    pqsignal(SIGTERM, die);
    BackgroundWorkerUnblockSignals();

    /* Connect to the database */
    /* Valid flags: BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION */
    BackgroundWorkerInitializeConnection(dbname, NULL, 0);

    /* Start a transaction to run our SPI commands */
    SetCurrentStatementStartTimestamp();
    StartTransactionCommand();
    PushActiveSnapshot(GetTransactionSnapshot());

    if (SPI_connect() != SPI_OK_CONNECT) {
        elog(ERROR, "db4ml_worker: could not connect to SPI manager");
    }

    /* * CRITICAL STEP: Call the Python function.
     * We use SQL to call the PL/Python function we defined in the .sql file.
     * This delegates all Python/Sklearn complexity to the PL/Python engine.
     */
    char sql_cmd[256];
    snprintf(sql_cmd, sizeof(sql_cmd), "SELECT db4ml.internal_run_job(%d)", job_id);

    int ret = SPI_execute(sql_cmd, false, 0);

    if (ret != SPI_OK_SELECT) {
        elog(WARNING, "db4ml_worker: Error calling internal_run_job (SPI Code %d)", ret);
    }

    SPI_finish();
    PopActiveSnapshot();
    CommitTransactionCommand();

    /* We are done. The process will exit now. */
    proc_exit(0);
}

/*
 * FRONTEND FUNCTION
 * User calls this: SELECT db4ml.train_async('SELECT * FROM iris', 'species');
 */
Datum db4ml_launch_training(PG_FUNCTION_ARGS) {
    text *sql_query_txt = PG_GETARG_TEXT_PP(0);
    text *target_col_txt = PG_GETARG_TEXT_PP(1);
    char *sql_query = text_to_cstring(sql_query_txt);
    char *target_col = text_to_cstring(target_col_txt);
    int64 job_id = 0;

    /* 1. Insert the job into db4ml.jobs to get an ID */
    SPI_connect();
    
    char insert_sql[1024];
    /* Using prepared statement args would be safer for production, simplified here */
    /* Note: quote_literal_cstr is internal, relying on SPI_execute_with_args is better */
    
    const char *cmd = "INSERT INTO db4ml.jobs(sql_query, target_column) VALUES($1, $2) RETURNING id";
    Oid argtypes[2] = { TEXTOID, TEXTOID };
    Datum values[2] = { CStringGetTextDatum(sql_query), CStringGetTextDatum(target_col) };
    
    if (SPI_execute_with_args(cmd, 2, argtypes, values, NULL, false, 1) != SPI_OK_INSERT_RETURNING) {
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to insert job into queue")));
    }
    
    /* Get the returned ID */
    bool isnull;
    job_id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    SPI_finish();

    /* 2. Launch the Dynamic Background Worker */
    BackgroundWorker worker;
    MemSet(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = BGW_NEVER_RESTART; /* If it crashes, don't retry automatically */
    
    /* We point to our C function library */
    sprintf(worker.bgw_library_name, "db4ml");
    sprintf(worker.bgw_function_name, "db4ml_worker_main");
    
    /* Name visible in pg_stat_activity */
    snprintf(worker.bgw_name, BGW_MAXLEN, "db4ml worker job %ld", job_id);
    
    /* PASS ARGUMENTS:
       bgw_main_arg: The Job ID (int32)
       bgw_extra: The Database Name (string) - Essential so worker connects to correct DB
    */
    worker.bgw_main_arg = Int32GetDatum((int32)job_id);
    
    /* Get current database name to pass to worker */
    char *dbname = get_database_name(MyDatabaseId);
    if (dbname) {
        snprintf(worker.bgw_extra, BGW_EXTRALEN, "%s", dbname);
    }

    BackgroundWorkerHandle *handle;
    if (!RegisterDynamicBackgroundWorker(&worker, &handle)) {
        ereport(ERROR, (errmsg("Failed to register dynamic background worker")));
    }

    /* We don't wait for it to finish. We return immediately. */
    PG_RETURN_INT64(job_id);
}