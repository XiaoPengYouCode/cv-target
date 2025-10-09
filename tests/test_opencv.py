import pytest
import allure


@allure.description("该测试预计会成功.")
def test_success():
    assert 1 + 1 == 2


@allure.description("该测试预计会成功.")
def test_another_success():
    assert "hello".upper() == "HELLO"


@allure.description("该测试预计会失败.")
def test_logic_failure():
    assert 2 * 2 == 5


@allure.description("该测试预计会失败.")
def test_failure():
    assert 1 + 1 == 3


@allure.description("该测试被故意跳过.")
def test_skipped():
    pytest.skip("Skipping this test for demonstration purposes.")


@allure.description("该测试引发了一个异常.")
def test_broken():
    raise Exception("This test is broken and raises an exception.")
