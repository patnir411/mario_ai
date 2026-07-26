from __future__ import annotations

import subprocess
import sys
import textwrap


def test_generic_adapter_search_render_import_without_nes_stack():
    """Shared GBA/GB paths must not import the mutually incompatible NES stack."""
    code = textwrap.dedent(
        """
        import sys

        # Model a base/GBA/SML environment even when this test runs in the NES venv.
        sys.modules["mario.env"] = None
        sys.modules["gym_super_mario_bros"] = None
        sys.modules["nes_py"] = None

        from mario.adapters import SMA4Adapter, SMLAdapter, SMB1Adapter
        from mario.entity_policy import EntityTransformer
        from mario.render import make_contact_sheet_adapter, replay
        from mario.search import beam_search_adapter, beam_search

        assert SMA4Adapter and SMLAdapter
        assert EntityTransformer and make_contact_sheet_adapter and beam_search_adapter

        for invoke in (
            lambda: SMB1Adapter(),
            lambda: replay(1, 1, 0, [], 8),
            lambda: beam_search(max_depth=0),
        ):
            try:
                invoke()
            except RuntimeError as exc:
                message = str(exc)
                assert ".[nes]" in message
            else:
                raise AssertionError("NES-only entry point did not explain the missing extra")
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
