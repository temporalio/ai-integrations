"""Span formatter copied from sdk-python for ADK test diagnostics."""

from collections.abc import Iterable

from opentelemetry.sdk.trace import ReadableSpan


def dump_spans(
    spans: Iterable[ReadableSpan],
    *,
    parent_id: int | None = None,
    with_attributes: bool = True,
    indent_depth: int = 0,
) -> list[str]:
    ret: list[str] = []
    for span in spans:
        if (not span.parent and parent_id is None) or (
            span.parent and span.parent.span_id == parent_id
        ):
            span_str = f"{'  ' * indent_depth}{span.name}"
            if with_attributes:
                span_str += f" (attributes: {dict(span.attributes or {})})"
            # Add links
            if span.links:
                span_links: list[str] = []
                for link in span.links:
                    for link_span in spans:
                        if (
                            link_span.context is not None
                            and link_span.context.span_id == link.context.span_id
                        ):
                            span_links.append(link_span.name)
                span_str += f" (links: {', '.join(span_links)})"
            # Signals can duplicate in rare situations, so we make sure not to
            # re-add
            if "Signal" in span_str and span_str in ret:
                continue
            ret.append(span_str)
            ret += dump_spans(
                spans,
                parent_id=span.context.span_id if span.context else None,
                with_attributes=with_attributes,
                indent_depth=indent_depth + 1,
            )
    return ret
