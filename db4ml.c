/*
 * db4ml.c – Linear regression (OLS) on a user-supplied SELECT.
 * Contract: the SELECT must return M columns of type double precision.
 * Columns 1..M-1 are features X. Column M is the target y.
 * We store artifact: {"intercept": b0, "weights":[b1..bp], "n_features": p}
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "catalog/pg_type.h"
#include "lib/stringinfo.h"
#include "utils/array.h"
#include "utils/lsyscache.h"   /* get_attname, get_typlenbyval, etc. */
#include "access/table.h"
#include <math.h>

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(db4ml_train_linear);
PG_FUNCTION_INFO_V1(db4ml_predict_linear);

/* --------- helpers --------- */

/* Solve A x = b in-place via Gaussian elimination with partial pivoting. A is n*n */
static bool gauss_solve(double *A, double *b, int n) {
  for (int k=0;k<n;k++) {
    int piv = k;
    double max = fabs(A[k*n + k]);
    for (int i=k+1;i<n;i++) {
      double v = fabs(A[i*n + k]);
      if (v > max) { max = v; piv = i; }
    }
    if (max < 1e-12) return false; /* singular */
    if (piv != k) {
      for (int j=k;j<n;j++) { double t=A[k*n+j]; A[k*n+j]=A[piv*n+j]; A[piv*n+j]=t; }
      double tb=b[k]; b[k]=b[piv]; b[piv]=tb;
    }
    double Akk = A[k*n+k];
    for (int i=k+1;i<n;i++) {
      double f = A[i*n+k]/Akk;
      if (f==0.0) continue;
      for (int j=k;j<n;j++) A[i*n+j] -= f*A[k*n+j];
      b[i] -= f*b[k];
    }
  }
  for (int i=n-1;i>=0;i--) {
    double s=b[i];
    for (int j=i+1;j<n;j++) s -= A[i*n+j]*b[j];
    double Aii = A[i*n+i];
    if (fabs(Aii) < 1e-12) return false;
    b[i]=s/Aii;
  }
  return true;
}

static double r2_score(const double *y, const double *yhat, int n) {
  if (n<=1) return 0.0;
  double mean=0.0; for (int i=0;i<n;i++) mean+=y[i]; mean/= (double)n;
  double sse=0.0, sst=0.0;
  for (int i=0;i<n;i++){ double e=y[i]-yhat[i]; sse+=e*e; double d=y[i]-mean; sst+=d*d; }
  if (sst <= 1e-18) return 0.0;
  return 1.0 - sse/sst;
}

static double predict_row(const double *beta, const double *x_no_intercept, int p) {
  double s = beta[0];
  for (int j=0;j<p;j++) s += beta[j+1]*x_no_intercept[j];
  return s;
}

/* --------- TRAIN: db4ml.train_linear(text sql, float8 test_ratio, jsonb params) -> bigint --------- */

Datum db4ml_train_linear(PG_FUNCTION_ARGS) {
  text  *qtxt       = PG_GETARG_TEXT_PP(0);
  double test_ratio = PG_GETARG_FLOAT8(1);
  Jsonb *params     = PG_GETARG_JSONB_P(2);

  if (test_ratio < 0.0 || test_ratio >= 1.0)
    ereport(ERROR, (errmsg("test_ratio must be in [0,1)")));

  char *sql = text_to_cstring(qtxt);

  if (SPI_connect() != SPI_OK_CONNECT)
    ereport(ERROR, (errmsg("SPI_connect failed")));

  /* Execute user SELECT read-only */
  int rc = SPI_execute(sql, true /* read-only */, 0);
  if (rc != SPI_OK_SELECT)
    ereport(ERROR, (errmsg("training SELECT failed")));

  TupleDesc tupdesc = SPI_tuptable->tupdesc;
  int m = tupdesc->natts;
  if (m < 2)
    ereport(ERROR, (errmsg("SELECT must return at least 2 columns (features..., target)")));

  /* Enforce all columns are float8 to keep conversions fast and safe */
  for (int j=1;j<=m;j++) {
    if (SPI_gettypeid(tupdesc, j) != FLOAT8OID)
      ereport(ERROR, (errmsg("column %d must be double precision; cast in your SELECT", j)));
  }

  int p = m - 1;             /* number of features */
  uint64 n = SPI_processed;  /* rows */
  if (n < (uint64)(p+1))
    ereport(ERROR, (errmsg("not enough rows: need at least p+1")));

  int d = p + 1; /* intercept + features */
  double *A = (double *) palloc0(sizeof(double)*d*d);
  double *b = (double *) palloc0(sizeof(double)*d);

  /* deterministic split head/tail */
  uint64 n_train = (uint64) floor((1.0 - test_ratio) * (double)n);
  if (n_train < (uint64)d) n_train = (uint64)d;
  if (n_train > n) n_train = n;
  uint64 n_test = n - n_train;

  /* accumulate normal equations on train rows */
  for (uint64 i=0;i<n_train;i++) {
    bool isnull=false;
    double *x_aug = (double *) palloc(sizeof(double)*d);
    x_aug[0]=1.0;
    for (int j=0;j<p;j++) {
      Datum dj = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, j+1, &isnull);
      if (isnull) { pfree(x_aug); ereport(ERROR,(errmsg("NULL in feature col %d at row %lu", j+1, (unsigned long)i))); }
      x_aug[j+1] = DatumGetFloat8(dj);
    }
    Datum dy = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, p+1, &isnull);
    if (isnull) { pfree(x_aug); ereport(ERROR,(errmsg("NULL in target y at row %lu", (unsigned long)i))); }
    double yv = DatumGetFloat8(dy);

    /* A += x x^T; b += x y */
    for (int r=0;r<d;r++) {
      b[r] += x_aug[r]*yv;
      for (int c=0;c<d;c++) A[r*d + c] += x_aug[r]*x_aug[c];
    }
    pfree(x_aug);
  }

  /* solve for beta */
  if (!gauss_solve(A, b, d))
    ereport(ERROR, (errmsg("normal equations are singular; features likely collinear")));
  double *beta = b;  /* beta[0]=intercept, beta[1..p]=weights */

  /* R^2 on train and test (second pass) */
  double r2_train = 0.0, r2_test = 0.0;
  if (n_train > 0) {
    double *yhat = (double *) palloc(sizeof(double)*n_train);
    double *y    = (double *) palloc(sizeof(double)*n_train);
    for (uint64 i=0;i<n_train;i++) {
      bool isnull=false;
      double *x_no = (double *) palloc(sizeof(double)*p);
      for (int j=0;j<p;j++) {
        Datum dj = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, j+1, &isnull);
        x_no[j] = DatumGetFloat8(dj);
      }
      Datum dy = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, p+1, &isnull);
      y[i] = DatumGetFloat8(dy);
      yhat[i] = predict_row(beta, x_no, p);
      pfree(x_no);
    }
    r2_train = r2_score(y, yhat, (int)n_train);
  }
  if (n_test > 0) {
    double *yhat = (double *) palloc(sizeof(double)*n_test);
    double *y    = (double *) palloc(sizeof(double)*n_test);
    for (uint64 ii=0; ii<n_test; ii++) {
      uint64 i = n_train + ii;
      bool isnull=false;
      double *x_no = (double *) palloc(sizeof(double)*p);
      for (int j=0;j<p;j++) {
        Datum dj = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, j+1, &isnull);
        x_no[j] = DatumGetFloat8(dj);
      }
      Datum dy = SPI_getbinval(SPI_tuptable->vals[i], tupdesc, p+1, &isnull);
      y[ii] = DatumGetFloat8(dy);
      yhat[ii] = predict_row(beta, x_no, p);
      pfree(x_no);
    }
    r2_test = r2_score(y, yhat, (int)n_test);
  }

  /* weights[] array (exclude intercept) */
  Datum *elems = (Datum *) palloc(sizeof(Datum)*p);
  for (int j=0;j<p;j++) elems[j] = Float8GetDatum(beta[j+1]);
  ArrayType *warr = construct_array(elems, p, FLOAT8OID, sizeof(float8), true, 'd');

  /* Insert model; artifact built in SQL via jsonb_build_object + to_jsonb(weights[]) */
  rc = SPI_execute_with_args(
    "INSERT INTO db4ml.models(kind, query_text, n_features, artifact, params) "
    "VALUES('linear', $1, $2, jsonb_build_object('intercept',$3,'weights',to_jsonb($4),'n_features',$2), '{}'::jsonb) "
    "RETURNING id",
    4,
    (Oid[]){TEXTOID, INT4OID, FLOAT8OID, FLOAT8ARRAYOID},
    (Datum[]){CStringGetTextDatum(sql), Int32GetDatum(p), Float8GetDatum(beta[0]), PointerGetDatum(warr)},
    NULL, false, 1);

  if (rc != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
    ereport(ERROR, (errmsg("INSERT into db4ml.models failed")));

  bool isnull=false;
  Datum id_d = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
  if (isnull) ereport(ERROR,(errmsg("NULL id from models insert")));
  int64 model_id = DatumGetInt64(id_d);

  /* Insert metrics */
  rc = SPI_execute_with_args(
    "INSERT INTO db4ml.metrics(model_id, train_rows, test_rows, train_r2, test_r2) "
    "VALUES($1,$2,$3,$4,$5)",
    5,
    (Oid[]){INT8OID, INT4OID, INT4OID, FLOAT8OID, FLOAT8OID},
    (Datum[]){Int64GetDatum(model_id), Int32GetDatum((int)n_train), Int32GetDatum((int)n_test),
              Float8GetDatum(r2_train), Float8GetDatum(r2_test)},
    NULL, false, 1);
  if (rc != SPI_OK_INSERT)
    ereport(ERROR,(errmsg("INSERT into db4ml.metrics failed")));

  SPI_finish();
  PG_RETURN_INT64(model_id);
}

/* --------- PREDICT: db4ml.predict_linear(model_id, features float8[]) -> float8 --------- */

Datum db4ml_predict_linear(PG_FUNCTION_ARGS) {
  int64 model_id = PG_GETARG_INT64(0);
  ArrayType *feat = PG_GETARG_ARRAYTYPE_P(1);

  if (ARR_NDIM(feat) != 1)
    ereport(ERROR,(errmsg("features must be a 1-D float8 array")));

  int nfeat = (int) ARR_DIMS(feat)[0];

  if (SPI_connect() != SPI_OK_CONNECT)
    ereport(ERROR,(errmsg("SPI_connect failed")));

  /* Fetch n_features, intercept, and weights[] from artifact */
  int rc = SPI_execute_with_args(
    "SELECT (artifact->>'n_features')::int, "
    "       (artifact->>'intercept')::float8, "
    "       (SELECT array_agg((w->>0)::float8 ORDER BY ord) "
    "          FROM jsonb_array_elements(artifact->'weights') WITH ORDINALITY AS t(w,ord)) "
    "FROM db4ml.models WHERE id = $1",
    1, (Oid[]){INT8OID}, (Datum[]){Int64GetDatum(model_id)}, NULL, true, 1);

  if (rc != SPI_OK_SELECT || SPI_processed != 1)
    ereport(ERROR,(errmsg("model not found")));

  bool isnull=false;
  Datum d_nf = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
  if (isnull) ereport(ERROR,(errmsg("artifact missing n_features")));
  int stored_p = DatumGetInt32(d_nf);
  if (stored_p != nfeat)
    ereport(ERROR,(errmsg("features length %d != model.n_features %d", nfeat, stored_p)));

  Datum d_b0 = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull);
  if (isnull) ereport(ERROR,(errmsg("artifact missing intercept")));
  double b0 = DatumGetFloat8(d_b0);

  Datum d_w = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull);
  if (isnull) ereport(ERROR,(errmsg("artifact missing weights")));
  ArrayType *warr = DatumGetArrayTypeP(d_w);
  if (ARR_NDIM(warr)!=1 || (int)ARR_DIMS(warr)[0]!=nfeat)
    ereport(ERROR,(errmsg("weights shape mismatch")));

  /* Compute dot product b0 + w.x using raw array data */
  double pred = b0;
  double *xdat = (double *) ARR_DATA_PTR(feat);
  double *wdat = (double *) ARR_DATA_PTR(warr);
  for (int j=0;j<nfeat;j++) pred += xdat[j]*wdat[j];

  SPI_finish();
  PG_RETURN_FLOAT8(pred);
}
