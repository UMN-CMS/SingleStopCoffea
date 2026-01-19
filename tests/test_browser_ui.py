import pytest
from textual.pilot import Pilot
from analyzer.cli.browser import ResultBrowser
from analyzer.core.results import ResultGroup, ResultBase
from attrs import define, field


# Mock ResultBase for testing
@define
class MockResult(ResultBase):
    def __iadd__(self, other):
        return self

    def iscale(self, value):
        return self

    def approxSize(self):
        return 0

    def finalize(self):
        return self


@pytest.mark.asyncio
async def test_browser_search():
    # Setup dummy results
    r1 = MockResult(name="Apple")
    r2 = MockResult(name="Banana")
    r3 = MockResult(name="Cherry")

    group = ResultGroup(name="Fruits", results={"a": r1, "b": r2})
    # Add a nested group
    subgroup = ResultGroup(name="Berries", results={"c": r3})
    group.addResult(subgroup)

    app = ResultBrowser(group)

    async with app.run_test() as pilot:
        # Check initial state: "Fruits" should be in the tree
        tree = app.query_one("ResultTree")
        assert str(tree.result_tree.root.children[0].label) == "Fruits"

        # Search for "Apple"
        await pilot.press("tab")  # Focus input if needed, or just type
        input_widget = app.query_one("Input")
        input_widget.value = "Apple"
        await pilot.pause()  # Wait for reactive update

        # Verify tree only shows Apple (and its parent Fruits)
        # implementation detail: tree.root -> Fruits -> Apple
        # Fruits should be visible because it contains Apple.

        # access the tree widget inside ResultTree
        inner_tree = tree.result_tree
        fruits_node = inner_tree.root.children[0]
        assert str(fruits_node.label) == "Fruits"
        # Apple is 'a' in the dict, but label is "Apple"
        # Wait, inside handleResult:
        # new_node = parent.add(result.name, data=result)
        # So label matches result.name ("Fruits")

        # Children of Fruits:
        # "Apple" (from r1), "Banana" (from r2), "Berries" (from subgroup)
        # With filter "Apple":
        # Apple matches. Banana does not. Berries (name doesn't match, child Cherry doesn't match) -> excluded?
        # Actually Cherry doesn't match "Apple".

        # Re-check logic:
        # def matches(res):
        #    if filter in res.name: return True
        #    if isinstance(ResultGroup): for child...

        # "Fruits" matches "Apple"? No.
        # But Fruits has child Apple. So matches(Fruits) returns True.
        # So Fruits added.
        # Inside Fruits:
        #   handleResult(FruitsNode, r1 (Apple)) -> matches? Yes. Added.
        #   handleResult(FruitsNode, r2 (Banana)) -> matches? No. Not added.
        #   handleResult(FruitsNode, subgroup (Berries)) -> matches?
        #       Berries name "Berries" != Apple.
        #       Berries children: Cherry != Apple.
        #       So matches(Berries) == False. Not added.

        assert len(fruits_node.children) == 1
        assert str(fruits_node.children[0].label) == "Apple"

        # Change search to "Berries"
        input_widget.value = "Berries"
        await pilot.pause()

        # Fruits should show (contains Berries)
        # Fruits -> Berries
        fruits_node = inner_tree.root.children[0]
        assert len(fruits_node.children) == 1
        assert str(fruits_node.children[0].label) == "Berries"

        # Berries should show children?
        # Logic:
        # should_add = matches(result)
        # matches("Berries") is True.
        # result is Group.
        # new_node = parent.add("Berries")
        # for child in result: handleResult(...)
        #   handleResult(BerriesNode, Cherry)
        #     matches(Cherry) ("Cherry" vs "Berries") -> False.
        #     So Cherry NOT added.

        # So Berries node exists but has no children shown?
        # This seems to be the current logic implementation.

        berries_node = fruits_node.children[0]
        assert len(berries_node.children) == 0
