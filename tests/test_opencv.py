import pytest
import allure


@allure.description("This test is expected to pass.")
def test_success():
    assert 1 + 1 == 2


@allure.description("This test is expected to pass as well.")
def test_another_success():
    assert "hello".upper() == "HELLO"


@allure.description("This test is expected to fail.")
def test_failure():
    assert 1 + 1 == 3


@allure.description("This test is skipped intentionally.")
def test_skipped():
    pytest.skip("Skipping this test for demonstration purposes.")


@allure.description("This test raises an exception.")
def test_broken():
    raise Exception("This test is broken and raises an exception.")
