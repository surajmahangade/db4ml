-- load_data.sql
-- Script to load training datasets for db4ml extension

\echo '=== Creating Tables ==='

-- Drop tables if they exist
DROP TABLE IF EXISTS housing_data CASCADE;
DROP TABLE IF EXISTS salary_data CASCADE;

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
    salary double precision    -- in dollars
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

\echo '=== Verifying Data ==='

-- Show row counts
SELECT 'Housing data rows:' AS dataset, COUNT(*)::text AS count FROM housing_data
UNION ALL
SELECT 'Salary data rows:' AS dataset, COUNT(*)::text AS count FROM salary_data;

-- Show sample data
\echo '\n--- Housing Data Sample ---'
SELECT * FROM housing_data LIMIT 5;

\echo '\n--- Salary Data Sample ---'
SELECT * FROM salary_data LIMIT 5;

\echo '\n=== Data Loading Complete ==='