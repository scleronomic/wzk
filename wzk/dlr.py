import os
import platform


PLATFORM_IS_LINUX = platform.system() == "Linux"

USERNAME = os.path.expanduser("~").split(sep="/")[-1]
TENH_JO = "tenh_jo"

LOCATION2USERNAME_DICT = dict(mac="jote",
                              dlr="tenh_jo",
                              gcp="johannes_tenhumberg_gmail_com")

USERNAME2LOCATION_DICT = dict(jote="mac",
                              tenh_jo="dlr",
                              baeuml="dlr",
                              bauml="dlr",
                              johannes_tenhumberg_gmail_com="gcp")


def where_am_i():
    try:
        location = USERNAME2LOCATION_DICT[USERNAME]
    except KeyError:
        location = "dlr"

    return location


LOCATION = where_am_i()

userstore = "/volume/USERSTORE"
homelocal = "/home_local"
home = "/home"


def __wrapper_user(user=None):
    if user is None:
        user = USERNAME
    return user


def get_userstore(user):
    user = __wrapper_user(user=user)
    return f"{userstore}/{user}"


def get_home(user):
    user = __wrapper_user(user=user)
    return f"{home}/{user}"


def get_homelocal(user, host=None):
    user = __wrapper_user(user=user)
    if host is None:
        return f"{homelocal}/{user}"
    else:
        return f"/net/{host}{homelocal}/{user}"


# Alternative storage places for the samples
USERSTORE = get_userstore(user=USERNAME)  # Daily Back-up, relies on connection -> not for large Measurements
HOMELOCAL = get_homelocal(user=USERNAME)  # No Back-up, but fastest drive -> use for calculation
HOME = get_home(user=USERNAME)
USB = f"/var/run/media/{TENH_JO}/DLR-MA"

USERSTORE_TENH = f"{userstore}/{TENH_JO}"

# Paper
if PLATFORM_IS_LINUX:
    DIR_PAPER = f"{USERSTORE}/paper"
else:
    DIR_PAPER = "/Users/jote/Documents/PhD/paper"

# CONFERENCES = dict(iros="iros",
#                    icra="icra",
#                    humanoids="humanoids",
#                    tro="tro")

Humanoids20_Calibration = DIR_PAPER + "/20humanoids_calibration"
Humanoids22_Calibration = DIR_PAPER + "/22humanoids_calibration"
IROS22_PLANNING = DIR_PAPER + "/22iros_planning"
IROS23_IK = DIR_PAPER + "/23iros_IK"
IROS23_Representation = "/23iros_Representation"
TRO23_Planning = "/23tro_Planning"

# Projects
__automatica22_dict = dict(dlr=f"{userstore}/tenh_jo/Automatica2022",
                           mac="/Users/jote/Documents/PhD/data/mogen/Automatica2022",
                           gcp="/home/johannes_tenhumberg_gmail_com/sdb/Automatica2022")
Automatica22 = __automatica22_dict[LOCATION]


# ----------------------------------------------------------------------------------------------------------------------
# DLR Remote Access

# VNC
# On remote PC (i.e. pandia):
#   vncpasswd  -> set password (optional)
#   vncserver -geometry 1680x1050 -depth 24 -> start server, with scaled resolution for mac
#
# On local PC (i.e. my laptop):
#   ssh -l tenh_jo -L 5901:pandia:5901 ssh.robotic.dlr.de  -> set link port to the remote display
#   open VNC and connect to home: 127.0.0.1:1

# Connect2
# ssh -D 8080 -l tenh_jo ssh.robotic.dlr.de
