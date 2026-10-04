"""Add V4 operation records and execution ownership."""

import sqlalchemy as sa
from alembic import op

revision = "0002_v4_operations"
down_revision = "0001_v3_control_plane"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(
            sa.Column("execution_generation", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True)))
        batch.drop_constraint("ck_jobs_kind", type_="check")
        batch.create_check_constraint(
            "ck_jobs_kind",
            "kind IN ('validate','build','update','evaluate','snapshot_verify','benchmark','backup','backup_verify','retention','diagnostics')",
        )
    op.create_table(
        "operations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("key", sa.String(255)),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("kind", "key", name="uq_operation_kind_key"),
    )
    op.create_index("ix_operations_kind", "operations", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_operations_kind", table_name="operations")
    op.drop_table("operations")
    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("heartbeat_at")
        batch.drop_column("execution_generation")
        batch.drop_constraint("ck_jobs_kind", type_="check")
        batch.create_check_constraint(
            "ck_jobs_kind",
            "kind IN ('validate','build','update','evaluate','snapshot_verify','benchmark')",
        )
