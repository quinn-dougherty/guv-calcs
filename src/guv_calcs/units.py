AVAILABLE_UNITS = ["meters", "feet"]


def meters_to_feet(arg):
    return arg / 0.3048


def feet_to_meters(arg):
    return arg * 0.3048


def convert(function, *args):
    result = tuple(function(arg) if arg is not None else None for arg in args)
    return result[0] if len(result) == 1 else result


def convert_units(src, dst, *args):
    """general purpose function for converting between units"""
    if not (isinstance(src, str) and isinstance(dst, str)):
        raise ValueError("Unit labels must be strings")

    src, dst = src.lower(), dst.lower()

    if src not in AVAILABLE_UNITS:
        raise ValueError(f"{src} is not an available unit")
    if dst not in AVAILABLE_UNITS:
        raise ValueError(f"{dst} is not an available unit")

    if src == dst:
        result = args
        if len(result) == 1:
            result = result[0]
    elif src == "meters":
        if dst == "feet":
            result = convert(meters_to_feet, *args)
    elif src == "feet":
        if dst == "meters":
            result = convert(feet_to_meters, *args)
    else:
        raise ValueError("Something went wrong that really should not have gone wrong")

    return result
