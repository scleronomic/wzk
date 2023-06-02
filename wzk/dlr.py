import os
import platform


PLATFORM_IS_LINUX = platform.system() == "Linux"

USERNAME = os.path.expanduser("~").split(sep="/")[-1]
TENH_JO = "tenh_jo"

userstore = "/volume/USERSTORE"
homelocal = "/home_local"
home = "/home"

MAC = "mac"
DLR = "dlr"
GCP = "gcp"


__LOCATION2USERNAME_DICT = {MAC: "jote",
                            DLR: "tenh_jo",
                            GCP: "johannes_tenhumberg_gmail_com"}
__USERNAME2LOCATION_DICT = {"jote": MAC,
                            "tenh_jo": DLR,
                            "baeuml": DLR,
                            "bauml": DLR,
                            "f_justin": DLR,
                            "johannes_tenhumberg_gmail_com": GCP}


# --- Functions --------------------------------------------------------------------------------------------------------
def where_am_i():
    try:
        location = __USERNAME2LOCATION_DICT[USERNAME]
    except KeyError:
        location = DLR

    return location


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


LOCATION = where_am_i()

USERSTORE = get_userstore(user=USERNAME)  # Daily Back-up, relies on connection -> not for large Measurements
HOMELOCAL = get_homelocal(user=USERNAME)  # No Back-up, but fastest drive -> use for calculation
HOME = get_home(user=USERNAME)
USB = f"/var/run/media/{TENH_JO}/DLR-MA"

USERSTORE_TENH = get_userstore(user=TENH_JO)

__DIR_BASE_DICT = {DLR: f"{USERSTORE_TENH}",
                   MAC: "/Users/jote/Documents/PhD"}
__DIR_BASE = __DIR_BASE_DICT[LOCATION]

# --- Data -------------------------------------------------------------------------------------------------------------
DIR_DATA = f"{__DIR_BASE}/data"
DIR_DATA_CAL = f"{DIR_DATA}/calibration"


# --- Paper ------------------------------------------------------------------------------------------------------------
DIR_PAPER = f"{__DIR_BASE}/paper"

Humanoids20_Calibration = DIR_PAPER + "/20humanoids_calibration"
Humanoids22_Calibration = DIR_PAPER + "/22humanoids_calibration"
IROS22_PLANNING = DIR_PAPER + "/22iros_planning"
IROS23_IK = DIR_PAPER + "/23iros_IK"
TRO23_Planning = "/23tro_Planning"
Humanoids23_Calibration = DIR_PAPER + "/23humanoids_calibration"


# -- Projects ----------------------------------------------------------------------------------------------------------
def get_dir_models(location, user):
    if user == "f_justin":
        directory = f"{USERSTORE}/motion_planning/models"
    else:
        __DIR_MODELS_DICT = {DLR: f"{USERSTORE_TENH}/data/models",
                             MAC: f"{__DIR_BASE_DICT[MAC]}/data/mogen/Automatica2022",
                             GCP: "/home/johannes_tenhumberg_gmail_com/sdb/Automatica2022"}
        directory = __DIR_MODELS_DICT[location]

    return directory


DIR_MODELS = get_dir_models(location=LOCATION, user=USERNAME)


# --- DLR Remote Access ------------------------------------------------------------------------------------------------

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
