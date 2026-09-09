"""Food calorie cache basis and migration tests."""

import os
import sys
import unittest

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import models
from app.food_cache import (
    LEGACY_UNKNOWN,
    PER_100G,
    PER_UNIT,
    build_calorie_basis,
    get_cached_calorie_reference,
    get_cached_total_calories,
    load_common_food_calorie_seeds,
    normalize_food_name,
    seed_common_food_calorie_cache,
    upsert_food_calorie_basis,
)
from scripts.migrate_phase02 import _replace_legacy_food_cache


class TestFoodCalorieCache(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        models.FoodCalorieCache.__table__.create(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_normalizes_name_and_unit_basis(self):
        self.assertEqual(normalize_food_name("  APPLE　Pie "), "apple pie")

        basis = build_calorie_basis("鸡蛋", 144, 2, "枚")
        self.assertEqual(basis.basis_type, PER_UNIT)
        self.assertEqual(basis.portion_unit, "个")
        self.assertEqual(basis.portion_qty, 1)
        self.assertEqual(basis.calories, 72)

    def test_weight_basis_is_stored_per_100g_and_scaled(self):
        basis = build_calorie_basis("鸡胸肉", 330, 200, "克")
        self.assertEqual(basis.basis_type, PER_100G)
        self.assertEqual(basis.portion_unit, "g")
        self.assertEqual(basis.calories, 165)

        upsert_food_calorie_basis(self.db, "鸡胸肉", 330, 200, "克", "api")
        self.db.commit()

        calories = get_cached_total_calories(self.db, " 鸡胸肉 ", 150, "g")
        self.assertEqual(calories, 247.5)

    def test_unit_basis_upserts_and_scales(self):
        upsert_food_calorie_basis(self.db, "鸡蛋", 144, 2, "个", "llm")
        self.db.commit()
        self.assertEqual(
            get_cached_total_calories(self.db, "鸡蛋", 3, "枚"),
            216,
        )

        upsert_food_calorie_basis(self.db, "鸡蛋", 75, 1, "个", "api")
        self.db.commit()

        rows = self.db.query(models.FoodCalorieCache).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].calories, 75)
        self.assertEqual(rows[0].source, "api")

        upsert_food_calorie_basis(self.db, "鸡蛋", 60, 1, "个", "llm")
        self.db.commit()
        rows = self.db.query(models.FoodCalorieCache).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].calories, 75)
        self.assertEqual(rows[0].source, "api")

    def test_different_units_have_independent_bases(self):
        upsert_food_calorie_basis(self.db, "苹果", 100, 1, "个", "llm")
        upsert_food_calorie_basis(self.db, "苹果", 70, 1, "份", "llm")
        self.db.commit()

        self.assertEqual(
            self.db.query(models.FoodCalorieCache).count(),
            2,
        )
        self.assertEqual(get_cached_total_calories(self.db, "苹果", 2, "个"), 200)
        self.assertEqual(get_cached_total_calories(self.db, "苹果", 2, "份"), 140)

    def test_legacy_rows_are_not_used(self):
        self.db.add(models.FoodCalorieCache(
            name="苹果",
            normalized_name="苹果",
            basis_type=LEGACY_UNKNOWN,
            portion_qty=1,
            portion_unit="个",
            calories=999,
            source="legacy",
        ))
        self.db.commit()

        self.assertIsNone(
            get_cached_total_calories(self.db, "苹果", 1, "个")
        )

    def test_common_food_seed_is_valid_and_idempotent(self):
        seeds = load_common_food_calorie_seeds()
        seeded_count = seed_common_food_calorie_cache(self.db)
        self.db.commit()

        self.assertEqual(seeded_count, len(seeds))
        self.assertEqual(
            self.db.query(models.FoodCalorieCache).count(),
            len(seeds),
        )
        self.assertEqual(
            get_cached_total_calories(self.db, "苹果", 200, "克"),
            104,
        )
        self.assertEqual(
            get_cached_total_calories(self.db, "苹果", 2, "个"),
            190,
        )
        reference = get_cached_calorie_reference(self.db, "苹果")
        self.assertEqual(reference["calories"], 52)
        self.assertEqual(reference["portion_qty"], 100)
        self.assertEqual(reference["portion_unit"], "g")

        seed_common_food_calorie_cache(self.db)
        self.db.commit()
        self.assertEqual(
            self.db.query(models.FoodCalorieCache).count(),
            len(seeds),
        )

        upsert_food_calorie_basis(
            self.db, "苹果", 60, 100, "g", "manual"
        )
        self.db.commit()
        seed_common_food_calorie_cache(self.db)
        self.db.commit()
        self.assertEqual(
            get_cached_total_calories(self.db, "苹果", 100, "g"),
            60,
        )


class TestFoodCacheMigration(unittest.TestCase):
    def test_archives_old_table_and_creates_empty_v2_table(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(text(
                """
                CREATE TABLE food_calorie_cache (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    portion_qty FLOAT,
                    portion_unit VARCHAR,
                    calories FLOAT NOT NULL,
                    source VARCHAR NOT NULL,
                    created_at TIMESTAMP
                )
                """
            ))
            connection.execute(text(
                """
                INSERT INTO food_calorie_cache
                    (id, name, portion_qty, portion_unit, calories, source)
                VALUES
                    (4, '苹果', 1, '份', 70, 'llm'),
                    (5, '苹果', 1, '个', 100, 'llm')
                """
            ))

            backup_name = _replace_legacy_food_cache(connection)

            self.assertIsNotNone(backup_name)
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("food_calorie_cache")
            }
            self.assertTrue(
                {"normalized_name", "basis_type", "updated_at"}.issubset(columns)
            )
            self.assertEqual(
                connection.execute(text(
                    "SELECT COUNT(*) FROM food_calorie_cache"
                )).scalar_one(),
                0,
            )
            self.assertEqual(
                connection.execute(text(
                    f'SELECT COUNT(*) FROM "{backup_name}"'
                )).scalar_one(),
                2,
            )

            connection.execute(text(
                """
                INSERT INTO food_calorie_cache
                    (name, normalized_name, basis_type, portion_qty,
                     portion_unit, calories, source)
                VALUES ('苹果', '苹果', 'per_unit', 1, '个', 95, 'api')
                """
            ))
            self.assertIsNone(_replace_legacy_food_cache(connection))
            self.assertEqual(
                connection.execute(text(
                    "SELECT COUNT(*) FROM food_calorie_cache"
                )).scalar_one(),
                1,
            )
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
