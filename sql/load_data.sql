-- load_data.sql
-- Script to load training datasets for db4ml extension

\echo '=== Creating Tables ==='

-- Drop tables if they exist
DROP TABLE IF EXISTS housing_data CASCADE;
DROP TABLE IF EXISTS salary_data CASCADE;
DROP TABLE IF EXISTS mobile_price_data CASCADE; -- ADDED
DROP TABLE IF EXISTS iris_data CASCADE;         -- ADDED

-- Create housing data table
CREATE TABLE housing_data (
    crim double precision,      -- per capita crime rate
    rm double precision,        -- average number of rooms
    age double precision,       -- proportion of units built before 1940
    dis double precision,       -- distance to employment centers
    tax double precision,       -- property tax rate
    ptratio double precision,   -- pupil-teacher ratio
    lstat double precision,     -- % lower status population
    medv double precision       -- median value (target, in $1000s)
);

-- Create salary data table
CREATE TABLE salary_data (
    years_experience double precision,
    salary double precision     -- in dollars
);

-- Create Mobile Price Classification table (from Kaggle 'train.csv')
-- Note: 'price_range' is the classification target (0, 1, 2, or 3)
CREATE TABLE mobile_price_data (
    battery_power INT,
    blue INT,
    clock_speed REAL,
    dual_sim REAL,
    fc INT,
    four_g INT,
    int_memory INT,
    m_dep REAL,
    mobile_wt REAL,
    n_cores INT,
    pc INT,
    px_height INT,
    px_width INT,
    ram INT,
    sc_h INT,
    sc_w INT,
    talk_time INT,
    three_g INT,
    touch_screen INT,
    wifi INT,
    price_range INT -- Target: 0 (low) to 3 (very high)
);

-- Create Iris Classification table
CREATE TABLE iris_data (
    sepal_length double precision,
    sepal_width double precision,
    petal_length double precision,
    petal_width double precision,
    species text -- Target: e.g., 'Iris-setosa'
);

\echo '=== Loading Housing Data ==='

-- Load housing data from CSV
COPY housing_data(crim, rm, age, dis, tax, ptratio, lstat, medv)
FROM '/src/db4ml/dataset/housing.csv'
DELIMITER ','
CSV HEADER;

\echo '=== Loading Salary Data ==='

-- Create temporary table for salary data (has extra index column)
CREATE TEMP TABLE salary_temp (
    idx int,
    years_experience double precision,
    salary double precision
);

-- Load all columns
COPY salary_temp FROM '/src/db4ml/dataset/Salary_dataset.csv'
DELIMITER ','
CSV HEADER;

-- Insert only the columns we need
INSERT INTO salary_data (years_experience, salary)
SELECT years_experience, salary FROM salary_temp;

-- Clean up temp table
DROP TABLE salary_temp;

\echo '=== Loading Mobile Price Data ==='

-- Load Mobile Price data from its training CSV
COPY mobile_price_data(
    battery_power, blue, clock_speed, dual_sim, fc, four_g, 
    int_memory, m_dep, mobile_wt, n_cores, pc, px_height, 
    px_width, ram, sc_h, sc_w, talk_time, three_g, 
    touch_screen, wifi, price_range
)
FROM '/src/db4ml/dataset/mobile_price_classify.csv' -- Assumes your file is named this
DELIMITER ','
CSV HEADER;

\echo '=== Loading Iris Data ==='

-- Load Iris data from its CSV
COPY iris_data(sepal_length, sepal_width, petal_length, petal_width, species)
FROM '/src/db4ml/dataset/iris_classify.csv' -- Assumes your file is named this
DELIMITER ','
CSV HEADER;


\echo '=== Verifying Data ==='

-- Show row counts
SELECT 'Housing data rows:' AS dataset, COUNT(*)::text AS count FROM housing_data
UNION ALL
SELECT 'Salary data rows:' AS dataset, COUNT(*)::text AS count FROM salary_data
UNION ALL
SELECT 'Mobile Price rows:' AS dataset, COUNT(*)::text AS count FROM mobile_price_data -- ADDED
UNION ALL
SELECT 'Iris data rows:' AS dataset, COUNT(*)::text AS count FROM iris_data;          -- ADDED

-- Show sample data
\echo '\n--- Mobile Price Data Sample ---'
SELECT battery_power, ram, price_range FROM mobile_price_data LIMIT 5;

\echo '\n--- Iris Data Sample ---'
SELECT * FROM iris_data LIMIT 5;

\echo '\n=== Data Loading Complete ==='