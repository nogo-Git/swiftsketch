import zlib

from PIL import Image, ImageDraw


PART_COLORS_RGBA = {
    "outline": (0.05, 0.05, 0.05, 1.0),
    "eye": (0.10, 0.30, 1.00, 1.0),
    "eyes": (0.10, 0.30, 1.00, 1.0),
    "nose": (1.00, 0.35, 0.05, 1.0),
    "mouth": (0.90, 0.00, 0.20, 1.0),
    "ear": (0.00, 0.65, 0.25, 1.0),
    "ears": (0.00, 0.65, 0.25, 1.0),
    "chest": (0.55, 0.20, 0.85, 1.0),
    "paw": (0.00, 0.70, 0.80, 1.0),
    "paws": (0.00, 0.70, 0.80, 1.0),
    "tail": (0.95, 0.65, 0.00, 1.0),
    "whisker": (0.75, 0.25, 0.05, 1.0),
    "whiskers": (0.75, 0.25, 0.05, 1.0),
}

FALLBACK_COLORS_RGBA = (
    (0.35, 0.55, 0.10, 1.0),
    (0.80, 0.20, 0.55, 1.0),
    (0.20, 0.55, 0.75, 1.0),
    (0.65, 0.40, 0.10, 1.0),
    (0.45, 0.25, 0.70, 1.0),
)


def semantic_part_rgba(part):
    normalized = str(part).strip().lower()
    if normalized in PART_COLORS_RGBA:
        return PART_COLORS_RGBA[normalized]

    color_index = zlib.crc32(normalized.encode("utf-8"))
    return FALLBACK_COLORS_RGBA[color_index % len(FALLBACK_COLORS_RGBA)]


def semantic_part_bgr(part):
    red, green, blue, _ = semantic_part_rgba(part)
    return tuple(round(channel * 255) for channel in (blue, green, red))


def semantic_part_rgb(part):
    red, green, blue, _ = semantic_part_rgba(part)
    return tuple(round(channel * 255) for channel in (red, green, blue))


def unique_semantic_parts(parts):
    return list(dict.fromkeys(str(part).strip().lower() for part in parts))


def add_semantic_legend(image, parts, panel_width=150):
    """Append a semantic-part color legend to the right edge of an image."""
    image = image.convert("RGB")
    parts = unique_semantic_parts(parts)
    row_height = 20
    top_margin = 30
    output_height = max(image.height, top_margin + row_height * len(parts) + 8)
    output = Image.new(
        "RGB",
        (image.width + panel_width, output_height),
        "white",
    )
    output.paste(image, (0, 0))

    draw = ImageDraw.Draw(output)
    legend_x = image.width + 12
    draw.text((legend_x, 8), "Semantic parts", fill=(25, 25, 25))

    for index, part in enumerate(parts):
        center_y = top_margin + index * row_height + row_height // 2
        color = semantic_part_rgb(part)
        draw.line(
            (legend_x, center_y, legend_x + 24, center_y),
            fill=color,
            width=4,
        )
        draw.text(
            (legend_x + 32, center_y - 7),
            part,
            fill=(25, 25, 25),
        )

    return output
