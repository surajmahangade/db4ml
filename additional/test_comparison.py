"""
Compare accuracy of fit() vs partial_fit() with PostgreSQL batch loading.
"""

import numpy as np
import psycopg2
from sklearn.datasets import load_postgres
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


def setup_database():
    """Setup test database with synthetic regression data."""

    # Connect to default database first to create new DB
    db_config = {
        "host": "127.0.0.1",  # force IPv4
        "port": 5432,  # integer
        "user": "dev",
        "password": "dev",
        "database": "dev",  # existing DB in Docker
    }

    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cursor = conn.cursor()

    # Drop and create database
    cursor.execute("DROP DATABASE IF EXISTS sklearn_comparison")
    cursor.execute("CREATE DATABASE sklearn_comparison")

    cursor.close()
    conn.close()

    # Connect to the new database
    db_config["database"] = "sklearn_comparison"
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # Create training table
    cursor.execute(
        """
        CREATE TABLE train_data (
            id SERIAL PRIMARY KEY,
            x1 FLOAT,
            x2 FLOAT,
            x3 FLOAT,
            x4 FLOAT,
            x5 FLOAT,
            target FLOAT
        )
        """
    )

    # Create test table
    cursor.execute(
        """
        CREATE TABLE test_data (
            id SERIAL PRIMARY KEY,
            x1 FLOAT,
            x2 FLOAT,
            x3 FLOAT,
            x4 FLOAT,
            x5 FLOAT,
            target FLOAT
        )
        """
    )

    print("Generating synthetic data...")
    print("  True model: target = 3*x1 + 2*x2 - 1.5*x3 + 0.5*x4 + 1*x5 + noise")

    np.random.seed(42)

    # Generate training data (100,000 samples)
    print("  Generating 10,000 training samples...")
    n_train = 10000
    for i in range(0, n_train, 1000):
        batch_size = min(1000, n_train - i)
        values = []

        for _ in range(batch_size):
            x1, x2, x3, x4, x5 = np.random.randn(5)
            target = (
                3 * x1 + 2 * x2 - 1.5 * x3 + 0.5 * x4 + 1 * x5 + np.random.randn() * 0.1
            )
            # Convert all to Python float to avoid np.float64 issue
            values.append(tuple(float(v) for v in (x1, x2, x3, x4, x5, target)))

        cursor.executemany(
            "INSERT INTO train_data (x1, x2, x3, x4, x5, target) VALUES (%s, %s, %s, %s, %s, %s)",
            values,
        )

        if (i + batch_size) % 10000 == 0:
            print(f"    Inserted {i + batch_size} training samples...")

    # Generate test data (10,000 samples)
    print("  Generating 1,000 test samples...")
    n_test = 1000
    values = []
    for _ in range(n_test):
        x1, x2, x3, x4, x5 = np.random.randn(5)
        target = (
            3 * x1 + 2 * x2 - 1.5 * x3 + 0.5 * x4 + 1 * x5 + np.random.randn() * 0.1
        )
        values.append(tuple(float(v) for v in (x1, x2, x3, x4, x5, target)))

    cursor.executemany(
        "INSERT INTO test_data (x1, x2, x3, x4, x5, target) VALUES (%s, %s, %s, %s, %s, %s)",
        values,
    )

    conn.commit()
    cursor.close()
    conn.close()

    print("Database setup complete!\n")
    return db_config


def test_fit_with_batching(db_config):
    """
    Approach 1: fit() with batch mode.
    Automatically detects batch/full-load mode.
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: fit() with BATCH MODE")
    print("=" * 70)

    train_data = load_postgres(
        query="SELECT x1, x2, x3, x4, x5, target FROM train_data ORDER BY id",
        db_config=db_config,
        target_column="target",
        batch_size=1000000,  # large enough to force full-load
        return_X_y=False,  # return Bunch
    )

    model = SGDRegressor(
        max_iter=1000,
        tol=1e-3,
        eta0=0.01,
        learning_rate="constant",
        random_state=42,
        verbose=0,
    )

    start_time = time.time()
    if train_data.batch_mode:
        # batch mode (unlikely here because batch_size > n_samples)
        first_batch = True
        for X_batch, y_batch in train_data.data:
            if first_batch:
                model.partial_fit(X_batch, y_batch)
                first_batch = False
            else:
                model.partial_fit(X_batch, y_batch)
    else:
        # full load
        X_train = train_data.data
        y_train = train_data.target
        model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Model coefficients: {model.coef_}")
    print(f"Expected coefficients: [3, 2, -1.5, 0.5, 1]")
    print(f"Model intercept: {model.intercept_[0]:.4f}")

    return model, training_time


def test_partial_fit_with_batching(db_config):
    """
    Approach 2: partial_fit() with batch mode.
    Automatically iterates over batches if batch mode is active.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: partial_fit() with BATCH MODE")
    print("=" * 70)

    train_data = load_postgres(
        query="SELECT x1, x2, x3, x4, x5, target FROM train_data ORDER BY id",
        db_config=db_config,
        target_column="target",
        batch_size=5000,  # smaller than dataset triggers batch mode
        return_X_y=False,
    )

    model = SGDRegressor(
        max_iter=1,  # partial_fit handles iteration
        tol=None,
        eta0=0.01,
        learning_rate="constant",
        random_state=42,
        verbose=0,
    )

    start_time = time.time()
    if train_data.batch_mode:
        # iterate batches from generator
        first_batch = True
        for X_batch, y_batch in train_data.data:
            if first_batch:
                model.partial_fit(X_batch, y_batch)
                first_batch = False
            else:
                model.partial_fit(X_batch, y_batch)
    else:
        # full load
        X_train = train_data.data
        y_train = train_data.target
        model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Model coefficients: {model.coef_}")
    print(f"Expected coefficients: [3, 2, -1.5, 0.5, 1]")
    print(f"Model intercept: {model.intercept_[0]:.4f}")

    return model, training_time


def evaluate_model(model, db_config, approach_name):
    """Evaluate model on test set."""

    print(f"\n{'='*70}")
    print(f"EVALUATION: {approach_name}")
    print(f"{'='*70}")

    # Load test data
    X_test, y_test = load_postgres(
        query="SELECT x1, x2, x3, x4, x5, target FROM test_data",
        db_config=db_config,
        batch_size=10000,
        target_column="target",
        return_X_y=True,
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest Set Metrics:")
    print(f"  Mean Squared Error (MSE):  {mse:.6f}")
    print(f"  Root Mean Squared Error:   {rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R² Score:                  {r2:.6f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def compare_results(results1, time1, results2, time2):
    """Compare results from both approaches."""

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'fit()':<20} {'partial_fit()':<20} {'Difference'}")
    print("-" * 70)

    print(
        f"{'Training Time (sec)':<25} {time1:<20.2f} {time2:<20.2f} {abs(time1-time2):.2f}"
    )
    print(
        f"{'MSE':<25} {results1['mse']:<20.6f} {results2['mse']:<20.6f} {abs(results1['mse']-results2['mse']):.6f}"
    )
    print(
        f"{'RMSE':<25} {results1['rmse']:<20.6f} {results2['rmse']:<20.6f} {abs(results1['rmse']-results2['rmse']):.6f}"
    )
    print(
        f"{'MAE':<25} {results1['mae']:<20.6f} {results2['mae']:<20.6f} {abs(results1['mae']-results2['mae']):.6f}"
    )
    print(
        f"{'R² Score':<25} {results1['r2']:<20.6f} {results2['r2']:<20.6f} {abs(results1['r2']-results2['r2']):.6f}"
    )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if abs(results1["r2"] - results2["r2"]) < 0.001:
        print("✓ Both approaches produce nearly identical accuracy!")
    else:
        print("! Accuracy differs between approaches")

    if abs(time1 - time2) < 1:
        print("✓ Both approaches have similar training time")
    elif time1 < time2:
        print(f"✓ fit() is faster by {time2-time1:.2f} seconds")
    else:
        print(f"✓ partial_fit() is faster by {time1-time2:.2f} seconds")


def main():
    """Main execution."""

    print("=" * 70)
    print("SKLEARN LINEAR REGRESSION: fit() vs partial_fit() COMPARISON")
    print("=" * 70)

    # Setup database
    db_config = setup_database()

    # Test Approach 1: fit() with batch mode
    model1, time1 = test_fit_with_batching(db_config)
    results1 = evaluate_model(model1, db_config, "fit() with batch mode")

    # Test Approach 2: partial_fit() with batch mode
    model2, time2 = test_partial_fit_with_batching(db_config)
    results2 = evaluate_model(model2, db_config, "partial_fit() with batch mode")

    # Compare
    compare_results(results1, time1, results2, time2)

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
