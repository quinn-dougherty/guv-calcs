import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path


def print_standard_zones(room):
    """
    display results of special calc zones
    """
    st.subheader("Efficacy", divider="grey")
    fluence = room.calc_zones["WholeRoomFluence"]
    if fluence.values is not None:
        avg_fluence = round(fluence.values.mean(), 3)
        fluence_str = ":blue[" + str(avg_fluence) + "] μW/cm2"
    else:
        fluence_str = None
    st.write("Average fluence: ", fluence_str)

    if fluence.values is not None:
        df = get_disinfection_table(avg_fluence, room)
        st.dataframe(df, hide_index=True)

    st.subheader("Indoor Air Chemistry", divider="grey")
    if fluence.values is not None:
        ozone_ppb = calculate_ozone_increase(room)
        ozone_color = "red" if ozone_ppb > 5 else "blue"
        ozone_str = f":{ozone_color}[**{round(ozone_ppb,2)} ppb**]"
    else:
        ozone_str = "Not available"
    st.write(f"Air changes from ventilation: **{room.air_changes}**")
    st.write(f"Ozone decay constant: **{room.ozone_decay_constant}**")
    st.write(f"Estimated increase in indoor ozone: {ozone_str}")

    st.subheader("Photobiological Safety", divider="grey")
    skin = room.calc_zones["SkinLimits"]
    eye = room.calc_zones["EyeLimits"]

    if skin.values is not None and eye.values is not None:
        hours_to_tlv, hours_skin, hours_eye = get_hours_to_tlv_room(room)
        SKIN_EXCEEDED = True if hours_skin < 8 else False
        EYE_EXCEEDED = True if hours_eye < 8 else False

        if hours_to_tlv > 8:
            hour_str = ":blue[Indefinite]"
        else:
            hour_str = f":red[{round(hours_to_tlv,2)}]"
        st.write(f"Hours before Threshold Limit Value is reached: **{hour_str}**")

    if skin.values is not None:
        skin_max = round(skin.values.max(), 3)
        color = "red" if SKIN_EXCEEDED else "blue"
        skin_str = "**:" + color + "[" + str(skin_max) + "]** " + skin.units
        if SKIN_EXCEEDED:
            skin_dim = round((hours_skin / 8) * 100, 1)
            skin_str += f" *(dimming required: {skin_dim}%)*"
    else:
        skin_str = None
    if eye.values is not None:
        eye_max = round(eye.values.max(), 3)
        color = "red" if EYE_EXCEEDED else "blue"
        eye_str = "**:" + color + "[" + str(eye_max) + "]** " + eye.units
        if EYE_EXCEEDED:
            eye_dim = round((hours_eye / 8) * 100, 1)
            eye_str += f" *(dimming required: {eye_dim}%)*"
    else:
        eye_str = None

    col_1, col_2 = st.columns(2)

    with col_1:
        st.write("Max Skin Dose (8 Hours): ", skin_str)
        if skin.values is not None:
            st.pyplot(skin.plot_plane(), **{"transparent": "True"})
        else:
            st.write("(Not available)")

    with col_2:
        st.write("Max Eye Dose (8 Hours): ", eye_str)
        if eye.values is not None:
            st.pyplot(eye.plot_plane(), **{"transparent": "True"})
        else:
            st.write("(Not available)")


def get_hours_to_tlv_room(room):
    """
    calculate the hours to tlv in a particular room, given a particular installation of lamps

    technically speaking; in the event of overlapping beams, it is possible to check which
    lamps are shining on that spot and what their spectra are. this function currently doesn't do that

    TODO: good lord this function is a nightmare. let's bust it up eventually.
    """

    # select standards
    if "ANSI IES RP 27.1-22" in room.standard:
        skin_standard = "ANSI IES RP 27.1-22 (Skin)"
        mono_skinmax = 479
        eye_standard = "ANSI IES RP 27.1-22 (Eye)"
        mono_eyemax = 161
    elif "IEC 62471-6:2022" in room.standard:
        skin_standard = "IEC 62471-6:2022 (Eye/Skin)"
        eye_standard = skin_standard
        mono_skinmax = 23
        mono_eyemax = mono_skinmax
    else:
        raise KeyError(f"Room standard {room.standard} is not valid")

    skin_limits = room.calc_zones["SkinLimits"]
    eye_limits = room.calc_zones["EyeLimits"]

    # iterate over all lamps
    hours_to_tlv_skin, hours_to_tlv_eye = [], []
    skin_maxes, eye_maxes = [], []
    for lamp_id, lamp in room.lamps.items():
        # get max irradiance shown by this lamp upon both zones
        skin_irradiance = lamp.max_irradiances[
            "SkinLimits"
        ]  # this will be in uW/cm2 no matter what
        eye_irradiance = lamp.max_irradiances["EyeLimits"]
        skin_maxes.append(skin_irradiance)
        eye_maxes.append(eye_irradiance)
        if len(lamp.spectra) > 0:
            # if lamp has a spectra associated with it, calculate the weighted spectra
            skin_hours = _get_weighted_hours_to_tlv(
                lamp, skin_irradiance, skin_standard
            )
            eye_hours = _get_weighted_hours_to_tlv(lamp, eye_irradiance, eye_standard)
        else:
            # if it doesn't, first, yell.
            st.warning(
                f"{lamp.name} does not have an associated spectra. Photobiological safety calculations will be inaccurate."
            )
            # then just use the monochromatic approximation
            skin_hours = mono_skinmax * 8 / skin_irradiance
            eye_hours = mono_eyemax * 8 / eye_irradiance
        hours_to_tlv_skin.append(skin_hours)
        hours_to_tlv_eye.append(eye_hours)

    # now check that overlapping beams in the calc zone aren't pushing you over the edge
    global_skin_max = skin_limits.values.max() / 3.6 / 8  # to uW/cm2
    global_eye_max = eye_limits.values.max() / 3.6 / 8

    if global_skin_max > max(skin_maxes) or global_eye_max > max(eye_maxes):
        # first pick a lamp to use the spectra of. one with a spectra is preferred.
        chosen_lamp = _select_representative_lamp(room, skin_standard)
        if len(chosen_lamp.spectra) > 0:
            # calculate weighted if possible
            new_hours_to_tlv_skin = _get_weighted_hours_to_tlv(
                chosen_lamp, global_skin_max, skin_standard
            )
            hours_to_tlv_skin.append(new_hours_to_tlv_skin)

            new_hours_to_tlv_eye = _get_weighted_hours_to_tlv(
                chosen_lamp, global_eye_max, eye_standard
            )
            hours_to_tlv_eye.append(new_hours_to_tlv_eye)
        else:
            hours_to_tlv_skin.append(
                mono_skinmax * 8 / skin_limits.values.max()
            )  # these will be in mJ/cm2/8 hrs
            hours_to_tlv_eye.append(mono_eyemax * 8 / eye_limits.values.max())

    # return the value of hours_to_tlv that will be most limiting
    all_hours_to_tlv = hours_to_tlv_skin + hours_to_tlv_eye
    hours_to_tlv = min(all_hours_to_tlv)

    return hours_to_tlv, min(hours_to_tlv_skin), min(hours_to_tlv_eye)


def _get_weighted_hours_to_tlv(lamp, irradiance, standard):
    """
    calculate hours to tlv for a particular lamp, calc zone, and standard
    """

    # get spectral data for this lamp
    wavelength = lamp.spectra["Unweighted"][0]
    rel_intensities = lamp.spectra["Unweighted"][1]
    # determine total power in the spectra as it corresponds to total power
    indices = np.intersect1d(
        np.argwhere(wavelength > 200), np.argwhere(wavelength < 280)
    )
    spectral_power = rel_intensities[indices].sum()
    ratio = irradiance / spectral_power
    power_distribution = (
        rel_intensities * ratio / 1000
    )  # to mJ/cm2 - this value is the "true" spectra at the calc zone level

    # load weights according to the standard
    weighting = lamp.spectral_weightings[standard][1]

    weighted_power = power_distribution * weighting

    seconds_to_tlv = 3 / _sum_spectrum(wavelength, weighted_power)
    hours_to_tlv = seconds_to_tlv / 3600
    return hours_to_tlv


def _sum_spectrum(wavelength, intensity):
    """
    sum across a spectrum
    """
    weighted_intensity = [
        intensity[i] * (wavelength[i] - wavelength[i - 1])
        for i in range(1, len(wavelength))
    ]
    return sum(weighted_intensity)


def _select_representative_lamp(room, standard):
    """
    select a lamp to use for calculating the spectral limits in the event
    that no single lamp is contributing exclusively to the TLVs
    """
    if len(set([lamp.filename for lamp_id, lamp in room.lamps.items()])) <= 1:
        # if they're all the same just use that one.
        chosen_lamp = room.lamps[room.lamps.keys()[0]]
    else:
        # otherwise pick the least convenient one
        weighted_sums = {}
        for lamp_id, lamp in room.lamps.items():
            # iterate through all lamps and pick the one with the highest value sum
            if len(lamp.spectra) > 0:
                # either eye or skin standard can be used for this purpose
                weighted_sums[lamp_id] = lamp.spectra[standard].sum()

        if len(weighted_sums) > 0:
            chosen_id = max(weighted_sums, key=weighted_sums.get)
            chosen_lamp = room.lamps[chosen_id]
        else:
            # if no lamps have a spectra then it doesn't matter. pick any lamp.
            chosen_lamp = room.lamps[room.lamps.keys()[0]]
    return chosen_lamp


def get_disinfection_table(fluence, room):

    """assumes all lamps are GUV222. in the future will need something cleverer than this"""

    wavelength = 222

    fname = Path("./data/disinfection_table.csv")
    df = pd.read_csv(fname)
    df = df[df["Medium"] == "Aerosol"]
    df = df[df["wavelength [nm]"] == wavelength]
    keys = ["Species", "Medium (specific)", "k [cm2/mJ]", "Ref", "Full Citation"]

    df = df[keys].fillna(" ")

    volume = room.get_volume()

    # convert to cubic feet for cfm
    if room.units == "meters":
        volume = volume / (0.3048 ** 3)

    df["eACH-UV"] = (df["k [cm2/mJ]"] * fluence * 3.6).round(2)
    df["CADR-UV [cfm]"] = (df["eACH-UV"] * volume / 60).round(2)
    df["CADR-UV [lps]"] = (df["CADR-UV [cfm]"] * 0.47195).round(2)
    df = df.rename(
        columns={"Medium (specific)": "Medium", "Full Citation": "Reference"}
    )

    newkeys = [
        "Species",
        # "Medium",
        "k [cm2/mJ]",
        "eACH-UV",
        "CADR-UV [cfm]",
        "CADR-UV [lps]",
        "Reference",
    ]
    df = df[newkeys]

    return df


def calculate_ozone_increase(room):
    """
    ozone generation constant is currently hardcoded to 10 for GUV222
    this should really be based on spectra instead
    but this is a relatively not very big deal, because
    """
    avg_fluence = room.calc_zones["WholeRoomFluence"].values.mean()
    ozone_gen = 10  # hardcoded for now, eventually should be based on spectra bu
    ach = room.air_changes
    ozone_decay = room.ozone_decay_constant
    ozone_increase = avg_fluence * ozone_gen / (ach + ozone_decay)
    return ozone_increase
