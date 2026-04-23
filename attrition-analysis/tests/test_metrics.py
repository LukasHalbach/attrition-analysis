import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "monthly_income": [3000.0, 5000.0, 4000.0, 6000.0, 7000.0, 9000.0],
            "job_satisfaction": [1, 1, 2, 2, 3, 3],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_correct_rates(sample_df):
    result = attrition_by_department(sample_df)
    # Each department has 2 employees and 1 leaver → 50% each
    assert (result["attrition_rate"] == 50.0).all()


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "department": ["Sales", "Sales", "HR", "HR", "HR"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_department(df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_correct_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_overtime(df)
    yes_rate = result.loc[result["overtime"] == "Yes", "attrition_rate"].iloc[0]
    no_rate = result.loc[result["overtime"] == "No", "attrition_rate"].iloc[0]
    assert yes_rate == 100.0
    assert no_rate == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_correct_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000.0, 5000.0, 7000.0, 9000.0],
        }
    )
    result = average_income_by_attrition(df)
    yes_avg = result.loc[result["attrition"] == "Yes", "avg_monthly_income"].iloc[0]
    no_avg = result.loc[result["attrition"] == "No", "avg_monthly_income"].iloc[0]
    assert yes_avg == 4000.0
    assert no_avg == 8000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_correct_attrition_rate():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [1, 1, 2, 2],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    sat1_rate = result.loc[result["job_satisfaction"] == 1, "attrition_rate"].iloc[0]
    sat2_rate = result.loc[result["job_satisfaction"] == 2, "attrition_rate"].iloc[0]
    assert sat1_rate == 100.0  # 2 leavers out of 2 employees
    assert sat2_rate == 0.0   # 0 leavers out of 2 employees


def test_satisfaction_summary_sorted_by_satisfaction_level():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [3, 1, 4, 2],
            "attrition": ["Yes", "No", "Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    levels = result["job_satisfaction"].tolist()
    assert levels == sorted(levels)
