"""Sanitize model SVG into the restricted subset used for coloring templates."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ALLOWED_TAGS = {"svg", "g", "rect", "path", "circle", "ellipse", "polyline", "polygon", "line"}
ALLOWED_ATTRS = {
    "svg": {"viewBox", "width", "height", "xmlns"},
    "g": {"stroke", "stroke-width", "fill", "stroke-linecap", "stroke-linejoin"},
    "rect": {"x", "y", "width", "height", "fill", "stroke", "stroke-width"},
    "path": {"d", "stroke", "stroke-width", "fill", "stroke-linecap", "stroke-linejoin"},
    "circle": {"cx", "cy", "r", "stroke", "stroke-width", "fill"},
    "ellipse": {"cx", "cy", "rx", "ry", "stroke", "stroke-width", "fill"},
    "polyline": {"points", "stroke", "stroke-width", "fill"},
    "polygon": {"points", "stroke", "stroke-width", "fill"},
    "line": {"x1", "y1", "x2", "y2", "stroke", "stroke-width"},
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _strip_comments(svg_text: str) -> str:
    return re.sub(r"<!--.*?-->", "", svg_text, flags=re.DOTALL)


def _sanitize_element(element: ET.Element, is_background: bool = False) -> ET.Element | None:
    tag = _local_name(element.tag)
    if tag not in ALLOWED_TAGS:
        return None

    clean = ET.Element(tag)
    allowed_attrs = ALLOWED_ATTRS[tag]
    for key, value in element.attrib.items():
        attr = _local_name(key)
        if attr in allowed_attrs and not attr.lower().startswith("on"):
            clean.set(attr, value)

    if tag == "svg":
        clean.set("viewBox", "0 0 2048 2048")
        clean.set("width", "2048")
        clean.set("height", "2048")
        clean.set("xmlns", SVG_NS)
    elif is_background and tag == "rect":
        clean.attrib.clear()
        clean.set("width", "2048")
        clean.set("height", "2048")
        clean.set("fill", "white")
    elif tag == "g":
        clean.set("stroke", "black")
        clean.set("fill", "none")
        clean.set("stroke-linecap", clean.get("stroke-linecap", "round"))
        clean.set("stroke-linejoin", clean.get("stroke-linejoin", "round"))
        if "stroke-width" not in clean.attrib:
            clean.set("stroke-width", "8")
    elif tag == "rect":
        clean.set("stroke", "black")
        clean.set("fill", "none")
        if "stroke-width" not in clean.attrib:
            clean.set("stroke-width", "4")
    elif tag != "rect":
        clean.set("stroke", "black")
        clean.set("fill", "none")

    for child in element:
        sanitized = _sanitize_element(child)
        if sanitized is not None:
            clean.append(sanitized)
    return clean


def sanitize_svg(svg_text: str) -> str:
    """Return a valid, restricted SVG document suitable for rasterization."""
    svg_text = _strip_comments(svg_text).strip()
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid SVG XML: {exc}") from exc

    if _local_name(root.tag) != "svg":
        raise ValueError("SVG root element is missing")

    clean_root = _sanitize_element(root)
    if clean_root is None:
        raise ValueError("SVG root element is not allowed")

    children = list(clean_root)
    has_background = bool(children and _local_name(children[0].tag) == "rect")
    if has_background:
        clean_root.remove(children[0])
        clean_root.insert(0, _sanitize_element(children[0], is_background=True))
    else:
        background = ET.Element("rect", width="2048", height="2048", fill="white")
        clean_root.insert(0, background)

    if len(clean_root) == 1:
        raise ValueError("SVG has no visible line art after sanitization")

    return ET.tostring(clean_root, encoding="unicode", short_empty_elements=True)
