"""
RMS python job to create new ERT observations for summary data with
localisation attributes.
Well position, well path info is extracted from RMS using xtgeo.
Dynamic observations are (temporarily) extracted from the old observation file
format for ERT observations for SUMMARY observations.
Influence range for elliptic shaped area and relevant zone(s) are
specified by the user as input in a config file to the script.

Output files: A csv file for observations and a csv file with localisation attribute
per well and obs type

Assumptions about input data:
Field parameter names are read from ERT config file (the FIELD keywords)
The zone names per field parameter is defined by using APS naming convention of field parameters where
zone name is part of the field parameter name.

"""

import copy
import math
from pathlib import Path
from typing import Any

import polars as pl
import xtgeo
import yaml

CONFIG_PATH = "/private/olia/IES_DL/"
FILENAME = "example_config_to_get_pos_and_loc_params_from_rms.yml"
CONFIG_FILE = CONFIG_PATH + FILENAME
PRJ = project  # RMS project


def read_yml_config(filename: str) -> dict:
    with open(filename, encoding="utf-8") as yml_file:
        return yaml.safe_load(yml_file)


def read_ert_summary_obs_file(filename: str) -> list[dict]:
    """
    Read old format ERT observation with SUMMARY observations
    Return a list of dict with observation data.
    """
    with open(filename, "r") as file:
        all_lines = file.readlines()
    line_number = 0
    obs_list = []
    for line in all_lines:
        line_number += 1
        # Remove endline
        line = line.strip()
        words = line.split()

        # Skip empty lines
        if len(words) == 0:
            continue

        # Skip comment lines
        if words[0].strip() == "--":
            continue

        # First word should be ERT key for summary observation
        ert_key = words[0].strip()
        if ert_key != "SUMMARY_OBSERVATION":
            continue

        # Now split the line at { and }
        words = line.split("{")
        string1 = words[0]
        string2 = words[1]

        # First part of string1 is ert keyword, second is ERT-ID for observation
        _, ert_id = string1.split()

        w = string2.split("}")
        # First part of string2 consists of all observation attributes
        obs_attributes_line = w[0]

        # Split the line of obs attributes into separate attributes
        # where each word consists of keyword=value
        obs_attributes_words = obs_attributes_line.split(";")
        obs_dict = {}
        for w2 in obs_attributes_words:
            w2 = w2.strip()
            if len(w2) == 0:
                continue
            # Split into keyword, '=' and value
            attribute_item = w2.split("=")
            if len(attribute_item) == 0:
                continue
            if len(attribute_item) != 2:
                raise ValueError(
                    f"Format error in file: {filename} for line number: {line_number}\n"
                )
            obs_attribute_key = attribute_item[0].strip().lower()
            obs_attribute_value = attribute_item[1].strip()
            obs_dict[obs_attribute_key] = obs_attribute_value

        # Get obs_type and wellname
        summary_vector = obs_dict["key"]
        obs_type, well_name = summary_vector.split(":")

        # Add additional info to obs
        obs_dict["ert_id"] = ert_id.strip()
        obs_dict["wellname"] = well_name.strip()
        obs_dict["obs_type"] = obs_type.strip()
        obs_dict["summary_vector"] = obs_dict["key"]
        obs_list.append(obs_dict)
        if len(obs_list) == 0:
            raise ValueError(f"No summary observations found in file: {filename}")
    return obs_list


def get_specification(spec: dict) -> tuple:
    """
    Check specification
    """
    kw_main = "localisation"
    if kw_main not in spec:
        raise KeyError(f"Missing keyword {kw_main} in {CONFIG_FILE}")
    local_dict = spec[kw_main]
    check_keywords_top_level(spec, kw_main)

    # Result output file name for summary obs in csv format
    result_summary_obs_file = get_file_name(
        local_dict, "result_summary_obs_file", kw_main
    )
    result_localisation_obs_file = get_file_name(
        local_dict, "result_localisation_obs_file", kw_main
    )

    expand_specification = get_expand_specification(local_dict, "expand_wildcards")

    # rms related settings
    use_well_head_position, grid_model_name, blocked_well_set_name, trajectory_name = (
        get_rms_settings(local_dict, "rms_settings", "localisation")
    )

    # input files with observations, renaming table,
    # field parameter names per zone,field correlation file
    (
        well_renaming_file,
        ert_obs_file,
        ert_config_field_param_file,
        rms_field_correlation_file,
    ) = get_input_files(local_dict, "input_files", kw_main)

    # Defined zone names and associated zone number
    zone_dict = get_zone_code_names(local_dict, "zone_codes", kw_main)

    default_field_settings = get_dict(
        local_dict, "default_field_settings", kw_main, required=True
    )
    default_ranges = get_ranges(
        default_field_settings, "ranges", "default_field_settings"
    )
    min_range_hwell = get_value(
        default_field_settings, "min_range_hwell", "default_field_settings"
    )
    mult_hwell_length = get_value(
        default_field_settings, "mult_hwell_length", "default_field_settings"
    )

    # Optional
    field_settings = get_dict(local_dict, "field_settings", kw_main, required=False)

    return (
        result_summary_obs_file,
        result_localisation_obs_file,
        use_well_head_position,
        grid_model_name,
        blocked_well_set_name,
        trajectory_name,
        well_renaming_file,
        ert_obs_file,
        ert_config_field_param_file,
        rms_field_correlation_file,
        default_ranges,
        min_range_hwell,
        mult_hwell_length,
        zone_dict,
        field_settings,
        expand_specification,
    )


def check_keywords_top_level(spec: dict, kw: str) -> None:
    keys = list(spec.keys())
    if len(keys) > 1:
        print("Keywords at top level:")
        print(f" {keys}")
        raise KeyError(f"Legal keyword: {kw}. Unknown keywords found")

    valid_keys = [
        "result_summary_obs_file",
        "result_localisation_obs_file",
        "expand_wildcards",
        "rms_settings",
        "input_files",
        "zone_codes",
        "default_field_settings",
        "field_settings",
    ]

    local_dict = spec[kw]
    keys = list(local_dict.keys())
    err_msg = []
    for key in keys:
        if key not in valid_keys:
            err_msg.append(key)
    if len(err_msg) > 0:
        print(f"Unknown keywords under {kw}:")
        for msg in err_msg:
            print(f"  {msg}")
        raise KeyError(f"Unknown keywords under keyword {kw}")


def get_file_name(input_dict: dict, kw: str, parent_kw: str) -> str:
    if kw not in input_dict:
        raise KeyError(f"Missing keyword {kw} under keyword {parent_kw}")
    return input_dict[kw]


def get_use_well_head_position(
    rms_settings_dict: dict, kw: str, default: bool = False
) -> bool:
    # Optional. Use well head position
    use_well_head_position = default
    if kw in rms_settings_dict:
        use_well_head_position = rms_settings_dict[kw]
    return use_well_head_position


def get_trajectory_name(
    rms_settings_dict: dict, kw: str, default: str = "Drilled trajectory"
) -> str:
    # Optional. Trajectory type as text string compatible with RMS wells
    trajectory_name = default
    if kw in rms_settings_dict:
        trajectory_name = rms_settings_dict[kw]
    return trajectory_name


def get_rms_settings(localisation_dict: dict, kw: str, parent_kw: str) -> tuple:
    """
    Check keyword rms_settings and return settings
    """
    if kw in localisation_dict:
        rms_settings_dict = localisation_dict[kw]
    else:
        raise ValueError(f"Missing keyword {kw} under keyword {parent_kw}")

    use_well_head_position = get_use_well_head_position(
        rms_settings_dict, "use_well_head_position"
    )

    # Optional if use well head position is True, otherwise required
    grid_model_name = None
    blocked_well_set_name = None
    if not use_well_head_position:
        grid_model_name = get_file_name(rms_settings_dict, "grid_model", kw)
        blocked_well_set_name = get_file_name(rms_settings_dict, "blocked_well_set", kw)

    trajectory_name = get_trajectory_name(rms_settings_dict, "trajectory")

    return (
        use_well_head_position,
        grid_model_name,
        blocked_well_set_name,
        trajectory_name,
    )


def get_input_files(localisation_dict: dict, kw: str, parent_kw: str) -> tuple:
    if kw not in localisation_dict:
        raise KeyError(
            f"Missing keyword {kw} in {CONFIG_FILE} under keyword {parent_kw}"
        )
    input_file_dict = localisation_dict[kw]
    well_renaming_table_file = get_file_name(input_file_dict, "well_renaming_table", kw)
    ert_obs_file = get_file_name(input_file_dict, "ert_summary_obs_file", kw)
    ert_config_field_param_file = get_file_name(
        input_file_dict, "ert_config_field_param_file", kw
    )
    rms_field_correlation_file = get_file_name(
        input_file_dict, "rms_field_correlation_file", kw
    )

    return (
        well_renaming_table_file,
        ert_obs_file,
        ert_config_field_param_file,
        rms_field_correlation_file,
    )


def get_dict(input_dict: dict, kw: str, parent_kw: str, required: bool = True) -> dict:
    if kw not in input_dict:
        if required:
            raise KeyError(f"Missing keyword {kw} under keyword {parent_kw}")
        return {}
    return input_dict[kw]


def get_ranges(input_dict: dict, kw: str, parent_kw: str) -> tuple:
    if kw not in input_dict:
        raise KeyError(f"Missing keyword {kw} under keyword {parent_kw}")
    ranges = input_dict[kw]
    if len(ranges) != 3:
        raise ValueError(
            "Expect 3 range parameters for influence ellipse: "
            "MainRange, PerpendicularRange, AnisotropyAngle."
            "Range parameters must be postive, but angle can also be 0."
            "Angle is expected to be in degrees measured "
            "from x-axis in anticlock direction."
        )
    if ranges[0] <= 0.0 or ranges[1] <= 0.0:
        raise ValueError("Expecting positive range parameters")
    return ranges


def get_value(input_dict: dict, kw: str, parent_kw: str):
    if kw not in input_dict:
        raise KeyError(f"Missing keyword {kw} under keyword {parent_kw}")
    value = float(input_dict[kw])
    if value <= 0.0:
        raise ValueError(
            f"Specified value for {kw} under  {parent_kw} must be positive"
        )
    return value


def read_renaming_table(filename: str) -> dict:
    with open(filename, "r") as file:
        all_lines = file.readlines()
    # Skip first two lines
    renaming_dict = {}
    for line in all_lines[2:]:
        words = line.split()
        rms_well_name = words[0]
        eclipse_well_name = words[1].strip()
        renaming_dict[eclipse_well_name] = rms_well_name.strip()
    return renaming_dict


def convert_to_rms_well_names(
    observation_list: list, renaming_dict: dict
) -> list[dict]:
    new_obs_list = []
    for obs_dict in observation_list:
        eclipse_well_name = obs_dict["wellname"]
        rms_well_name = renaming_dict.get(eclipse_well_name)
        if rms_well_name is None:
            raise KeyError(
                f"The eclipse well name: {eclipse_well_name} "
                f"is not defined in the renaming table. "
                "Cannot get RMS well name for this eclipse well"
            )
        new_obs_dict = copy.deepcopy(obs_dict)

        # Replace wellname
        new_obs_dict["wellname"] = rms_well_name
        new_obs_dict["sim_well_name"] = eclipse_well_name
        new_obs_list.append(new_obs_dict)
    return new_obs_list


def get_obs_types(obs_dict_list: list[dict]) -> list[str]:
    obs_types = []
    for obs_dict in obs_dict_list:
        obs_type = obs_dict["obs_type"]
        if obs_type not in obs_types:
            obs_types.append(obs_type)
    return obs_types


def get_well_names(obs_dict_list: list[dict]) -> list[str]:
    well_names = []
    for obs_dict in obs_dict_list:
        wellname = obs_dict["wellname"]
        if wellname not in well_names:
            well_names.append(wellname)
    return well_names


def get_position_of_well_observations(
    project,
    observation_list: list[dict],
    grid_model: Any,
    blocked_well_set: str = "BW",
    selected_wells: list = [],
    trajectory: str = "Drilled trajectory",
    use_well_head_position: bool = False,
) -> list[dict]:
    if use_well_head_position:
        print("Use well HEAD position as position of summary observations.")
        """Get well HEAD position and modify input observations by adding position"""
        use_selected_list = len(selected_wells) > 0
        for obs_dict in observation_list:
            wellname = obs_dict["wellname"]
            if not use_selected_list or wellname in selected_wells:
                well = xtgeo.well_from_roxar(project, wellname, trajectory=trajectory)
                obs_dict["xpos"] = well.xpos
                obs_dict["ypos"] = well.ypos
                obs_dict["hlength"] = 0.0
                obs_dict["well_path_angle"] = 0.0
    else:
        # Calculate mean position along blocked well and length of horizontal well path
        print(
            f"Use average well position from blocked well set: {blocked_well_set} "
            f"from grid model: {grid_model}"
        )
        for obs_dict in observation_list:
            wellname = obs_dict["wellname"]
            wellnames = [wellname]
            mean_position_dict = calculate_average_position_of_well_path(
                project, wellnames, grid_model, bw_name=blocked_well_set
            )
            obs_dict["xpos"] = mean_position_dict[wellname]["xpos"]
            obs_dict["ypos"] = mean_position_dict[wellname]["ypos"]
            obs_dict["hlength"] = mean_position_dict[wellname]["hlength"]
            obs_dict["well_path_angle"] = mean_position_dict[wellname]["angle"]

    return observation_list


def calculate_average_position_of_well_path(
    project,
    well_names: list[str],
    grid_model_name: str,
    bw_name: str = "BW",
    zone_log_name: str = "Zone",
    min_length: float = 150.0,
) -> dict[str, dict]:
    mean_position_dict = {}
    for wname in well_names:
        well = xtgeo.blockedwell_from_roxar(
            project, grid_model_name, bw_name, wname, lognames=[zone_log_name]
        )
        well.create_relative_hlen()
        df = well.get_dataframe()
        mean_position_dict[wname] = {
            "xpos": df["X_UTME"].mean(axis=0),
            "ypos": df["Y_UTMN"].mean(axis=0),
        }

        nrows = df.shape[0]
        horizontal_length = math.fabs(df.at[nrows - 1, "R_HLEN"])
        if horizontal_length < min_length:
            horizontal_length = 0.0
            rotation_angle = 0.0
        else:
            x_start_pos = df.at[0, "X_UTME"]
            y_start_pos = df.at[0, "Y_UTMN"]
            x_end_pos = df.at[nrows - 1, "X_UTME"]
            y_end_pos = df.at[nrows - 1, "Y_UTMN"]
            delta_x = math.fabs(x_end_pos - x_start_pos)
            delta_y = math.fabs(y_end_pos - y_start_pos)

            if delta_y > min_length:
                if delta_x > min_length:
                    # In degrees
                    rotation_angle = 180.0 * math.atan(delta_y / delta_x) / math.pi
                else:
                    rotation_angle = 90.0
            else:
                if delta_x > min_length:
                    rotation_angle = 0.0
        mean_position_dict[wname]["hlength"] = horizontal_length
        mean_position_dict[wname]["angle"] = rotation_angle
    return mean_position_dict


def read_field_param_names(ert_config_field_param_file: str) -> list[str]:
    with open(ert_config_field_param_file, "r") as file:
        all_lines = file.readlines()
    field_names = []
    for line in all_lines:
        words = line.split()
        if len(words) == 0:
            continue
        w = words[0].strip()
        if w[:2] == "--":
            continue
        if w.upper() == "FIELD":
            field_name = words[1].strip()
            field_names.append(field_name)
    return field_names


def get_zone_code_names(local_dict: dict, kw: str, parent_kw: str) -> dict:
    if kw not in local_dict:
        raise KeyError(f"Missing keyword {kw} under keyword {parent_kw}")
    return local_dict[kw]


def get_defined_zone_names(input_field_names: list[str], zone_dict: dict) -> list[str]:
    check_field_names(input_field_names, zone_dict)
    used_zone_names = []
    for name, value in zone_dict.items():
        zone_name = name.strip()
        for field_name in input_field_names:
            if (zone_name in field_name) and (zone_name not in used_zone_names):
                used_zone_names.append(zone_name)
    if len(used_zone_names) == 0:
        print("Field names:")
        print(f"{input_field_names=}")
        print("Zone names and zone numbers defined:")
        print(f"{zone_dict=}")
        raise ValueError("No field names belongs to the specified defined zone names ")
    return used_zone_names


def check_field_names(input_field_names: list[str], zone_dict: dict) -> None:
    err_list = []
    for field_name in input_field_names:
        found = False
        for name, value in zone_dict.items():
            if name in field_name:
                found = True
        if not found:
            err_list.append(field_name)
    if len(err_list) > 0:
        print("Field names which does not belong to any of the specified zones:")
        for name in err_list:
            print(f"  {name}")
        raise ValueError(
            "Field parameter names should contain the zone name as "
            "part of the field name."
        )


def get_string_array(input_dict: dict, kw: str, parent_kw: str) -> list[str]:
    if kw not in input_dict:
        raise KeyError(f"Missing keyword {kw} under keyword {parent_kw}")
    return input_dict[kw]


def get_expand_specification(local_dict: dict, kw: str) -> bool:
    expand = True
    if kw in local_dict:
        expand = local_dict[kw]
    return expand


def expand_wildcards(
    patterns: list[str], list_of_words: list[str], err_msg: str
) -> set[str]:
    all_matches = []
    errors = []
    for pattern in patterns:
        matches = [words for words in list_of_words if Path(words).match(pattern)]
        if len(matches) > 0:
            all_matches.extend(matches)
        else:
            errors.append(f"No match for: {pattern}")
    all_matches_set = set(all_matches)
    if len(errors) > 0:
        raise ValueError(f" {err_msg}\n     {errors}, available: {list_of_words}")
    return all_matches_set


def check_specified_strings(
    string_list: list[str],
    defined_string_list: list[str],
    string_name: str,
    config_file_name: str,
) -> None:
    err_strings = []
    err = 0
    for value in string_list:
        if value not in defined_string_list:
            err += 1
            err_strings.append(value)
    if err > 0:
        print(f"Errors: Unknown {string_name} is specified in {config_file_name}")
        for s in err_strings:
            print(f"  {s}")
        raise ValueError(f"Unknown {string_name}")


def write_result_summary_obs(
    filename: str,
    all_obs_dict: dict,
    allow_overwrite: bool = False,
    use_polars=True,
) -> None:
    """
    Write csv file with following columns:
    - summary_vector
    - date
    - obs_value
    - obs_error
    - min_error
    - max_error
    - zone_name
    """
    filepath = Path(filename)
    if filepath.exists() and not allow_overwrite:
        raise IOError(
            f"The file {filename} already exists. "
            "Choose another filename to write ERT summary observations."
        )
    print(f"Write file:  {filename}")

    summary_vector_header = "SUMMARY_KEY"
    date_header = "DATE"
    value_header = "VALUE"
    error_header = "ERROR"
    min_error_header = "MIN_ERROR"
    max_error_header = "MAX_ERROR"
    zone_name_header = "ZONE"

    if not use_polars:
        max_summary_vector_length = 12
        for key, obs_dict in all_obs_dict.items():
            (zone_name, ert_id) = key
            summary_vector = obs_dict["summary_vector"]
            if max_summary_vector_length < len(summary_vector):
                max_summary_vector_length = len(summary_vector)

        max_summary_vector_length += 2

        max_dates_length = 12
        max_value_length = 12
        max_zone_name_length = 12
        with open(filename, "w") as file:
            # Heading
            content = ""
            content += f"{summary_vector_header:<{max_summary_vector_length}}"
            content += f"{date_header:<{max_dates_length}}"
            content += f"{value_header:>{max_value_length}}"
            content += f"{error_header:>{max_value_length}}"
            content += f"{min_error_header:>{max_value_length}}"
            content += f"{max_error_header:>{max_value_length}}"
            content += f"{zone_name_header:>{max_zone_name_length}}"
            content += "\n"
            file.write(content)

            for key, obs_dict in all_obs_dict.items():
                (zone_name, ert_id) = key
                summary_vector = obs_dict["summary_vector"]
                value = float(obs_dict["value"])
                error = float(obs_dict["error"])
                date = obs_dict["date"]
                min_error = 0.5 * error
                max_error = 1.5 * error
                content = ""
                content += f"{summary_vector:<{max_summary_vector_length}}"
                content += f"{date:<{max_dates_length}}"
                content += f"{value:{max_value_length}.2f}"
                content += f"{error:{max_value_length}.2f}"
                content += f"{min_error:{max_value_length}.2f}"
                content += f"{max_error:{max_value_length}.2f}"
                content += f"{zone_name:>{max_zone_name_length}}"
                content += "\n"
                file.write(content)
    else:
        summary_vector = []
        value_list = []
        error_list = []
        date_list = []
        zone_list = []
        ert_id_list = []
        min_error_list = []
        max_error_list = []
        for key, obs_dict in all_obs_dict.items():
            (zone_name, ert_id) = key
            summary_vector.append(obs_dict["summary_vector"])
            value_list.append(float(obs_dict["value"]))
            error = float(obs_dict["error"])
            error_list.append(error)
            min_error_list.append(error * 0.5)  # TODO What should this be?
            max_error_list.append(error * 1.5)  # TODO What should this be?
            date_list.append(obs_dict["date"])
            zone_list.append(zone_name)
            ert_id_list.append(ert_id)
        data_dict = {
            "ert_id": ert_id_list,
            "summary_vector": summary_vector,
            "date": date_list,
            "value": value_list,
            "error": error_list,
            "min_error": min_error_list,
            "max_error": max_error_list,
            "zone_name": zone_list,
        }
        df = pl.DataFrame(data_dict)
        df.write_csv(filename, separator=" ")


def write_localisation_obs_attributes(
    filename: str,
    all_obs_dict: dict,
    allow_overwrite: bool = False,
    use_polars=True,
) -> None:
    """
    Write csv file with following columns:
    - summary_vector
    - xpos
    - ypos
    - range1
    - range2
    - angle
    - zone_name
    """

    filepath = Path(filename)
    if filepath.exists() and not allow_overwrite:
        raise IOError(
            f"The file {filename} already exists. "
            "Choose another filename to write ERT summary observations."
        )
    print(f"Write file:  {filename}")

    summary_vector_header = "SUMMARY_KEY"
    zone_name_header = "ZONE"
    xpos_header = "XPOS"
    ypos_header = "YPOS"
    main_range_header = "MAIN_RANGE"
    perp_range_header = "PERP_RANGE"
    anisotropy_angle_header = "ANISOTROPY_ANGLE"
    if not use_polars:
        max_summary_vector_length = 12
        max_zone_name_length = 12
        max_range_length = 12
        max_angle_length = 12
        for key, obs_dict in all_obs_dict.items():
            (zone_name, ert_id) = key
            if max_zone_name_length < len(zone_name):
                max_zone_name_length = len(zone_name)

        max_summary_vector_length += 2
        max_zone_name_length += 2
        max_value_length = 12
        with open(filename, "w") as file:
            # Heading
            content = ""
            content += f"{summary_vector_header:<{max_summary_vector_length}}"
            content += f"{xpos_header:>{max_value_length}}"
            content += f"{ypos_header:>{max_value_length}}"
            content += f"{main_range_header:>{max_range_length}}"
            content += f"{perp_range_header:>{max_range_length}}"
            content += f"{anisotropy_angle_header:>{max_angle_length}}"
            content += f"{zone_name_header:>{max_zone_name_length}}"
            content += "\n"
            file.write(content)

            localisation_param_written = {}
            for key, obs_dict in all_obs_dict.items():
                (zone_name, ert_id) = key
                summary_vector = obs_dict["summary_vector"]
                key_written = (zone_name, summary_vector)
                if key_written in localisation_param_written:
                    continue

                xpos = float(obs_dict["xpos"])
                ypos = float(obs_dict["ypos"])
                main_range = float(obs_dict["main_range"])
                perp_range = float(obs_dict["perp_range"])
                anisotropy_angle = float(obs_dict["anisotropy_angle"])
                content = ""
                content += f"{summary_vector:<{max_summary_vector_length}}"
                content += f"{xpos:{max_value_length}.1f}"
                content += f"{ypos:{max_value_length}.1f}"
                content += f"{main_range:{max_range_length}.1f}"
                content += f"{perp_range:{max_range_length}.1f}"
                content += f"{anisotropy_angle:{max_angle_length}.1f}"
                content += f"{zone_name:>{max_zone_name_length}}"
                content += "\n"
                file.write(content)

                localisation_param_written[key_written] = True
    else:
        summary_vector = []
        xpos_list = []
        ypos_list = []
        main_range_list = []
        perp_range_list = []
        anisotropy_angle_list = []
        zone_list = []
        ert_id_list = []

        for key, obs_dict in all_obs_dict.items():
            (zone_name, ert_id) = key

            summary_vector.append(obs_dict["summary_vector"])
            xpos_list.append(float(obs_dict["xpos"]))
            ypos_list.append(float(obs_dict["ypos"]))
            main_range_list.append(float(obs_dict["main_range"]))
            perp_range_list.append(float(obs_dict["perp_range"]))
            anisotropy_angle_list.append(float(obs_dict["anisotropy_angle"]))
            zone_list.append(zone_name)
            ert_id_list.append(ert_id)
        data_dict = {
            "ert_id": ert_id_list,
            "summary_vector": summary_vector,
            "xpos": xpos_list,
            "ypos": ypos_list,
            "main_range": main_range_list,
            "perp_range": perp_range_list,
            "anisotropy_angle": anisotropy_angle_list,
            "zone_name": zone_list,
        }
        df = pl.DataFrame(data_dict)
        df.write_csv(filename, separator=" ")


def write_obs_with_localization(
    filename: str,
    all_obs_dict: dict,
    allow_overwrite: bool = False,
) -> None:
    """
    Write csv file with following columns:
    - observation_key
    - summary_vector
    - date
    - obs_value
    - obs_error
    - min_error
    - max_error
    - xpos
    - ypos
    - range1
    - range2
    - angle
    - zone_name
    """

    filepath = Path(filename)
    if filepath.exists() and not allow_overwrite:
        raise IOError(
            f"The file {filename} already exists. "
            "Choose another filename to write ERT summary observations."
        )
    print(f"Write file:  {filename}")
    ert_id_list = []
    summary_vector = []
    date_list = []
    value_list = []
    error_list = []
    min_error_list = []
    max_error_list = []
    xpos_list = []
    ypos_list = []
    main_range_list = []
    perp_range_list = []
    anisotropy_angle_list = []
    zone_list = []

    for key, obs_dict in all_obs_dict.items():
        (zone_name, ert_id) = key
        ert_id_list.append(ert_id)
        summary_vector.append(obs_dict["summary_vector"])
        date_list.append(obs_dict["date"])
        value_list.append(float(obs_dict["value"]))
        error = float(obs_dict["error"])
        error_list.append(error)
        min_error_list.append(error * 0.5)  # TODO What should this be?
        max_error_list.append(error * 1.5)  # TODO What should this be?
        xpos_list.append(float(obs_dict["xpos"]))
        ypos_list.append(float(obs_dict["ypos"]))
        main_range_list.append(float(obs_dict["main_range"]))
        perp_range_list.append(float(obs_dict["perp_range"]))
        anisotropy_angle_list.append(float(obs_dict["anisotropy_angle"]))
        zone_list.append(zone_name)

        data_dict = {
            "observation_key": ert_id_list,
            "summary_vector": summary_vector,
            "date": date_list,
            "value": value_list,
            "error": error_list,
            "min_error": min_error_list,
            "max_error": max_error_list,
            "xpos": xpos_list,
            "ypos": ypos_list,
            "main_range": main_range_list,
            "perp_range": perp_range_list,
            "anisotropy_angle": anisotropy_angle_list,
            "zone_name": zone_list,
        }

    df = pl.DataFrame(data_dict)
    df.write_csv(filename, separator=" ")


def create_obs_local(project, config_file):
    """
    Read config file for influence range for distance based localisation for each
    well, observation type, zone
    Read position data for wells from RMS project
    Read production observation data (SUMMARY observation) from old ERT observation file
    Write a csv file with info related to:
    - localisation influence range for each well,
    - observation type,
    - zone
    """

    # Read config file defining localisation influence range for
    # each observation of summary type
    if not Path(config_file).exists():
        raise IOError("No such file:" + config_file)
    print(f"Read file: {config_file}")
    spec = read_yml_config(config_file)

    (
        result_summary_obs_file,
        result_localisation_obs_file,
        use_well_head_position,
        grid_model_name,
        blocked_well_set_name,
        trajectory_name,
        well_renaming_table_file,
        obs_summary_file,
        field_param_config_file,
        rms_field_correlation_file,
        default_ranges,
        min_range_hwell,
        mult_hwell_length,
        zone_dict,
        field_settings_spec_list,
        expand_specification,
    ) = get_specification(spec)
    
    print(f"Read file: {obs_summary_file}")
    obs_dict_list = read_ert_summary_obs_file(obs_summary_file)

    print(f"Read file: {well_renaming_table_file}")
    renaming_dict = read_renaming_table(well_renaming_table_file)

    new_obs_dict_list = convert_to_rms_well_names(obs_dict_list, renaming_dict)
    defined_obs_types = get_obs_types(new_obs_dict_list)
    defined_well_names = get_well_names(new_obs_dict_list)

    # Get position of observations from RMS
    print("Get well positions from RMS")
    new_obs_dict_list = get_position_of_well_observations(
        project,
        new_obs_dict_list,
        grid_model_name,
        blocked_well_set=blocked_well_set_name,
        trajectory=trajectory_name,
        use_well_head_position=use_well_head_position,
    )

    print(f"Read file:  {field_param_config_file}")
    defined_field_names = read_field_param_names(field_param_config_file)
    defined_zone_names = get_defined_zone_names(defined_field_names, zone_dict)

    # Set default settings for all observations initially
    output_dict_default = {}
    output_dict = {}
    for zone_name in defined_zone_names:
        for obs_dict in new_obs_dict_list:
            obs_localisation_dict = {}
            ert_id = obs_dict["ert_id"]
            result_id = (zone_name, ert_id)
            obs_localisation_dict = copy.deepcopy(obs_dict)
            obs_localisation_dict["main_range"] = default_ranges[0]
            obs_localisation_dict["perp_range"] = default_ranges[1]
            obs_localisation_dict["anisotropy_angle"] = default_ranges[2]
            obs_localisation_dict["summary_vector"] = obs_dict["summary_vector"]
            if obs_localisation_dict["hlength"] > min_range_hwell:
                well_path_angle = obs_localisation_dict["well_path_angle"]
                # Localisation ellipse main axis in same
                # direction as horizontal well path
                obs_localisation_dict["anisotropy_angle"] = well_path_angle
                obs_localisation_dict["main_range"] = max(
                    obs_localisation_dict["hlength"],
                    mult_hwell_length * obs_localisation_dict["main_range"],
                )

            if result_id not in output_dict_default:
                output_dict_default[result_id] = obs_localisation_dict

    # Optional keyword 'field_settings'
    if field_settings_spec_list:
        # Update field settings for specified fields, wells, obs_types
        print("Start field settings")
        for field_settings in field_settings_spec_list:
            zone_names = get_string_array(field_settings, "zone_name", "field_settings")
            if len(zone_names) == 1 and zone_names[0].lower() == "all":
                # Use all fields for all zones
                zone_names = defined_zone_names
            if expand_specification:
                zone_names = expand_wildcards(
                    zone_names,
                    defined_zone_names,
                    "Cannot expand wildcard notation of zone names "
                    "to any defined zone names specified in FIELD keyword in ERT.",
                )

            obs_types = get_string_array(field_settings, "obs_type", "field_settings")
            if len(obs_types) == 1 and obs_types[0].lower() == "all":
                obs_types = defined_obs_types
            if expand_specification:
                obs_types = expand_wildcards(
                    obs_types,
                    defined_obs_types,
                    "Cannot expand wildcard notation of observation types "
                    "to any defined observation types defined in "
                    f"observation file {obs_summary_file}",
                )

            well_names = get_string_array(
                field_settings, "well_names", "field_settings"
            )
            if len(well_names) == 1 and well_names[0].lower() == "all":
                well_names = defined_well_names
            if expand_specification:
                well_names = expand_wildcards(
                    well_names,
                    defined_well_names,
                    "Cannot expand wildcard notation of wellnames "
                    "to any defined well name in RMS model",
                )
            ranges = get_ranges(field_settings, "ranges", "field_settings")

            check_specified_strings(
                zone_names, defined_zone_names, "zone names", config_file
            )

            check_specified_strings(
                obs_types, defined_obs_types, "observation types", config_file
            )

            check_specified_strings(
                well_names, defined_well_names, "well names", config_file
            )

            for zone_name in zone_names:
                for obs_dict in new_obs_dict_list:
                    if (
                        obs_dict["obs_type"] in obs_types
                        and obs_dict["wellname"] in well_names
                    ):
                        ert_id = obs_dict["ert_id"]
                        result_id = (zone_name, ert_id)
                        if result_id not in output_dict:
                            #                            print(f"Add   {result_id}")
                            obs_localisation_dict = {}
                            obs_localisation_dict = copy.deepcopy(obs_dict)
                            obs_localisation_dict["main_range"] = ranges[0]
                            obs_localisation_dict["perp_range"] = ranges[1]
                            obs_localisation_dict["anisotropy_angle"] = ranges[2]
                            obs_localisation_dict["summary_vector"] = obs_dict[
                                "summary_vector"
                            ]
                            output_dict[result_id] = obs_localisation_dict

    # All observations not specified under field_settings is
    # added with default settings
    for key, obs_localisation_dict in output_dict_default.items():
        if key not in output_dict:
            output_dict[key] = obs_localisation_dict

    # Write result
    write_result_summary_obs(
        result_summary_obs_file, output_dict, allow_overwrite=True, use_polars=True
    )
    write_obs_with_localization(
        result_localisation_obs_file, output_dict, allow_overwrite=True
    )


if __name__ == "__main__":
    create_obs_local(PRJ, CONFIG_FILE)
    print("Finished")
