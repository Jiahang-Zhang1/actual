from __future__ import annotations


def apply_sklearn_compat_patches() -> None:
    """Best-effort patches so the recovered sklearn artifact remains loadable
    even if a slightly newer sklearn runtime is used.
    """
    try:
        import sklearn.compose._column_transformer as column_transformer
        if not hasattr(column_transformer, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            column_transformer._RemainderColsList = _RemainderColsList
    except Exception:
        pass
