"""Database migration utilities."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import asyncpg
from alembic.config import Config
from alembic import command


async def run_migrations(direction: str = "up", database_url: str = None):
    """Run database migrations."""
    if database_url is None:
        database_url = "postgresql://feast:feast@localhost:5432/ai_risk_manager"
    
    if direction == "up":
        await upgrade(database_url)
    elif direction == "down":
        await downgrade(database_url)
    else:
        raise ValueError(f"Unknown direction: {direction}")


async def upgrade(database_url: str):
    """Run upgrade migrations."""
    print("Running upgrade migrations...")
    
    # Alembic upgrade
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")
    
    print("Upgrade complete")


async def downgrade(database_url: str):
    """Run downgrade migrations."""
    print("Running downgrade migrations...")
    
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.downgrade(alembic_cfg, "-1")
    
    print("Downgrade complete")


async def create_tables(database_url: str):
    """Create initial tables if they don't exist."""
    conn = await asyncpg.connect(database_url)
    
    try:
        # Create core tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS merchants (
                merchant_id VARCHAR(50) PRIMARY KEY,
                category VARCHAR(50),
                gmv_30d BIGINT,
                txn_count_30d INTEGER,
                chargeback_rate_30d DOUBLE PRECISION,
                refund_rate_30d DOUBLE PRECISION,
                avg_order_value DOUBLE PRECISION,
                churn_risk_score DOUBLE PRECISION,
                onboarding_date TIMESTAMP,
                risk_tier VARCHAR(20),
                fraud_rate_30d DOUBLE PRECISION,
                dispute_win_rate_90d DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id VARCHAR(50) PRIMARY KEY,
                txn_count_30d INTEGER,
                avg_txn_amount DOUBLE PRECISION,
                return_rate_90d DOUBLE PRECISION,
                chargeback_count_90d INTEGER,
                device_count_30d INTEGER,
                upi_handle_count_30d INTEGER,
                first_txn_date TIMESTAMP,
                risk_tier VARCHAR(20),
                lifetime_value DOUBLE PRECISION,
                preferred_categories JSONB,
                preferred_payment_methods JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_fingerprint VARCHAR(100) PRIMARY KEY,
                os VARCHAR(20),
                model VARCHAR(50),
                sensor_entropy DOUBLE PRECISION,
                app_tamper_score DOUBLE PRECISION,
                emulator_score DOUBLE PRECISION,
                root_score DOUBLE PRECISION,
                vpn_probability DOUBLE PRECISION,
                first_seen TIMESTAMP,
                merchant_count_30d INTEGER,
                customer_count_30d INTEGER,
                upi_handle_count_30d INTEGER,
                fraud_reports_30d INTEGER,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                order_id VARCHAR(50) PRIMARY KEY,
                merchant_id VARCHAR(50) REFERENCES merchants(merchant_id),
                customer_id VARCHAR(50) REFERENCES customers(customer_id),
                device_fingerprint VARCHAR(100) REFERENCES devices(device_fingerprint),
                amount_paise BIGINT,
                category VARCHAR(50),
                payment_method VARCHAR(20),
                shipping_address_hash VARCHAR(32),
                billing_address_hash VARCHAR(32),
                timestamp TIMESTAMP,
                delivery_status VARCHAR(20),
                return_status VARCHAR(20),
                chargeback_status VARCHAR(20),
                is_fraud BOOLEAN,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id BIGSERIAL PRIMARY KEY,
                request_id VARCHAR(50),
                merchant_id VARCHAR(50),
                vector VARCHAR(20),
                risk_score DOUBLE PRECISION,
                action VARCHAR(20),
                model_version VARCHAR(50),
                features JSONB,
                shap_values JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_labels (
                id BIGSERIAL PRIMARY KEY,
                request_id VARCHAR(50),
                vector VARCHAR(20),
                true_label INTEGER,
                label_source VARCHAR(50),
                confidence DOUBLE PRECISION,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_merchant_time 
            ON transactions(merchant_id, timestamp)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_customer_time 
            ON transactions(customer_id, timestamp)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_predictions_merchant_time 
            ON model_predictions(merchant_id, created_at)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_labels_request 
            ON feedback_labels(request_id)
        """)
        
        print("Tables created successfully")
        
    finally:
        await conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate.py [up|down|create]")
        sys.exit(1)
    
    direction = sys.argv[1]
    database_url = sys.argv[2] if len(sys.argv) > 2 else None
    
    if direction == "create":
        asyncio.run(create_tables(database_url))
    else:
        asyncio.run(run_migrations(direction, database_url))


if __name__ == "__main__":
    main()