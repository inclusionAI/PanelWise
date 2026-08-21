"""Independent answers -> evaluation -> one synthesized answer."""

import asyncio

from panelwise import PanelWise


async def main() -> None:
    async with PanelWise.from_yaml("panelwise.yaml", eval=True) as engine:
        result = await engine.run(
            "Compare the strongest arguments for and against carbon taxes.",
            request_id="example-eval",
        )
    if not result.ok:
        raise RuntimeError("; ".join(result.errors))
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
