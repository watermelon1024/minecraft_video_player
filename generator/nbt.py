import nbtlib
from nbtlib.tag import Byte, Compound, Double, Int, List, String


def convert_to_nbt(value):
    """
    Parses JSON-like structures and converts them to NBT structure.
    """
    if isinstance(value, dict):
        return Compound({k: convert_to_nbt(v) for k, v in value.items()})
    elif isinstance(value, list):
        return List[Compound]([convert_to_nbt(v) for v in value])
    elif isinstance(value, str):
        return String(value)
    elif isinstance(value, int):
        return Int(value)
    elif isinstance(value, float):
        return Double(value)
    elif isinstance(value, bool):
        return Byte(1 if value else 0)
    else:
        raise ValueError(f"Unsupported data type: {type(value)}")


def create_frame_structure(filename: str, raw_json, data_version: int = 3465):
    """
    filename: Output filename (e.g., "frame_001.nbt")
    raw_json: The JSON pixel text data
    data_version: Minecraft version number (3465 is 1.20.1; important for compatibility)
    """

    # 1. Define Marker Entity
    # This is the carrier for our data
    marker_entity = Compound(
        {
            "pos": List[Double]([0.5, 0.5, 0.5]),  # Entity at center of structure
            "blockPos": List[Int]([0, 0, 0]),  # Corresponding block coordinates
            "nbt": Compound(
                {
                    "id": String("minecraft:marker"),  # Entity type
                    "Tags": List[String](["video_player", "frame_data"]),
                    # --- Key Part ---
                    # We define a custom "frame" tag on the marker to store text
                    "data": Compound({"frame": List(convert_to_nbt(raw_json))}),
                    # ---------------
                }
            ),
        }
    )

    # 2. Define Structure File (Structure Format)
    # Standard format for Minecraft structure blocks, must include size, entities, blocks, palette, etc.
    structure_file = Compound(
        {
            "size": List[Int]([1, 1, 1]),  # Structure size 1x1x1
            "entities": List[Compound]([marker_entity]),
            "blocks": List[Compound]([]),  # No blocks needed, so empty
            "palette": List[Compound](
                [Compound({"Name": String("minecraft:air")})]  # Empty palette definition is usually required
            ),
            "DataVersion": Int(data_version),  # This line is very important!
        }
    )

    # 3. Save File (Must use Gzip compression)
    file = nbtlib.File(structure_file)
    file.save(filename, gzipped=True)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    # --- Usage Example ---
    raw_json = [{"text": ""}, {"text": "█", "color": "#FF0000"}, {"text": "█", "color": "#00FF00"}]

    # Generate frame001.nbt
    create_frame_structure("frame_001.nbt", raw_json)
