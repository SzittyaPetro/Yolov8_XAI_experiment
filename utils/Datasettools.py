import json
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any

class CityClasses(Enum):
    """
    Enum class for Cityscapes classes.
    """
    PERSON = 0
    RIDER = 1
    BICYCLE = 2
    CAR = 3
    MOTORCYCLE = 4
    BUS = 5
    ONRAILS = 6
    TRUCK = 7
    EGO_VEHICLE = 8





class YoloClasses(Enum):
    """
    Enum class for YOLO classes.
    """
    small_vehicle = 0
    person = 1
    large_vehicle = 2
    two_wheeler = 3
    On_rails = 4

    def __str__(self):
        return self.name
    def __int__(self):
        return self.value

    @staticmethod
    def from_CityClasses(city_class):
        """
        Convert CityClasses to YoloClasses.
        Returns
        -------

        """
        if city_class == CityClasses.PERSON.value or city_class == CityClasses.RIDER.value:
            return YoloClasses.person
        elif city_class == CityClasses.MOTORCYCLE.value or city_class == CityClasses.BICYCLE.value:
            return YoloClasses.two_wheeler
        elif city_class == CityClasses.CAR.value or city_class == CityClasses.EGO_VEHICLE.value:
            return YoloClasses.small_vehicle
        elif city_class == CityClasses.BUS.value or city_class == CityClasses.TRUCK.value:
            return YoloClasses.large_vehicle
        elif city_class == CityClasses.ONRAILS.value:
            return YoloClasses.On_rails
        else:
            return 254


project_root = Path(__file__).resolve().parent

def convert_to_coco_table(cityscapes_data: Dict[str, Any]) -> List[str]:
    """
    Convert Cityscapes data to COCO format.

    This function processes Cityscapes data and converts it into COCO format annotations.

    Parameters
    ----------
    cityscapes_data : dict
        Dictionary containing Cityscapes data.

    Returns
    -------
    list of str
        List of COCO format annotations as strings.
    """

    coco_table = []

    for obj in cityscapes_data['objects']:
        label = obj['label']
        polygons = obj['polygon']

        min_x = min(coord[0] for coord in polygons)
        min_y = min(coord[1] for coord in polygons)
        max_x = max(coord[0] for coord in polygons)
        max_y = max(coord[1] for coord in polygons)

        width = (max_x - min_x) / 2048
        height = (max_y - min_y) / 1024
        pos_x = (min_x + max_x) / (2 * 2048)
        pos_y = (min_y + max_y) / (2 * 1024)

        item_id = next((clas.value for clas in CityClasses if label.lower().replace(" ","_").find(clas.name.lower()) != -1), 255)
        item_id = obj.get('id', item_id)

        if item_id != 255:
            coco_table.append(f"{int(YoloClasses.from_CityClasses(item_id))} {pos_x} {pos_y} {width} {height}")

    return coco_table

def convert_cityscapes_to_coco(item: Path) -> None:
    """
        Convert a Cityscapes JSON file to COCO format and save it.

        This function reads a Cityscapes JSON file, converts the data to COCO format, and saves the result to a text file.

        Parameters
        ----------
        item : Path
            Path to the Cityscapes JSON file.

        Returns
        -------
        None
    """
    if item.suffix != '.json':
        return

    with open(item, 'r') as f:
        cityscapes_data = json.load(f)
        coco_data = convert_to_coco_table(cityscapes_data)

    city = item.parent.stem
    action = item.parent.parent.stem

    coco_path = project_root.parent/"data" / "gtFine" / 'labels' / action / city
    coco_path.mkdir(parents=True, exist_ok=True)

    temp = '_'.join(item.stem.split('_')[0:3])
    coco_filepath = coco_path / f"{temp}_leftImg8bit.txt"

    with open(coco_filepath, 'w') as f:
        f.write('\n'.join(coco_data) + '\n')

    txt = project_root.parent/"data/gtFine" / f"{action}.txt"
    with open(txt, 'a') as w:
        w.write(f"{project_root.parent}/data/gtFine/images/{action}/{city}/{coco_filepath.stem}.png\n")

def process_folders(root_path: Path) -> None:
    """
    Process all Cityscapes JSON files in a directory.

    This function recursively processes all Cityscapes JSON files in a given directory, converting them to COCO format.

    Parameters
    ----------
    root_path : Path
        Root directory containing Cityscapes JSON files.

    Returns
    -------
    None
    """
    for action in root_path.iterdir():
        if action.is_dir():
            for city in action.iterdir():
                if city.is_dir():
                    for item in city.rglob('*.json'):
                        convert_cityscapes_to_coco(item)


# Example usage:
process_folders(project_root.parent/ 'data/gtFine/images_natur')