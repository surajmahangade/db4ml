EXTENSION = db4ml
MODULE_big = db4ml
OBJS = db4ml.o
DATA = sql/db4ml--0.2.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
