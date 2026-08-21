"""Panel proposals -> one shared shell action -> repeat."""

import asyncio

from panelwise import PanelWise


async def main() -> None:
    async with PanelWise.from_yaml(
        "panelwise.yaml",
        eval=False,
        workspace=".",
    ) as engine:
        result = await engine.run(
            "Add input validation to the public configuration loader and run its focused tests.",
            request_id="example-trajectory",
        )
    print(result.output)
    if patch := result.artifacts.get("patch"):
        print("\nGenerated patch:\n", patch)


if __name__ == "__main__":
    asyncio.run(main())
